"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _real_token_states(layer_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    real_idx = torch.nonzero(attention_mask > 0, as_tuple=False).squeeze(-1)
    if real_idx.numel() == 0:
        return layer_states[:1]
    return layer_states[real_idx]


def _selected_layer_indices(n_layers: int) -> list[int]:
    anchors = [max(0, n_layers // 4), max(0, n_layers // 2), max(0, (3 * n_layers) // 4), n_layers - 1]
    unique_sorted = sorted(set(anchors))
    return unique_sorted


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    n_layers, _, hidden_dim = hidden_states.shape
    layer_ids = _selected_layer_indices(n_layers)

    pooled_parts: list[torch.Tensor] = []
    for layer_id in layer_ids:
        token_states = _real_token_states(hidden_states[layer_id], attention_mask)
        mean_pool = token_states.mean(dim=0)
        max_pool = token_states.max(dim=0).values
        last_pool = token_states[-1]
        pooled_parts.extend([mean_pool, max_pool, last_pool])

    final_layer_tokens = _real_token_states(hidden_states[-1], attention_mask)
    tail_len = max(1, int(0.4 * final_layer_tokens.size(0)))
    answer_tail_mean = final_layer_tokens[-tail_len:].mean(dim=0)
    pooled_parts.append(answer_tail_mean)

    feature = torch.cat(pooled_parts, dim=0)

    if feature.numel() == 0:
        return torch.zeros(hidden_dim, dtype=hidden_states.dtype)
    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    n_layers, seq_len, _ = hidden_states.shape
    layer_ids = _selected_layer_indices(n_layers)
    final_tokens = _real_token_states(hidden_states[-1], attention_mask)
    n_real = int(final_tokens.size(0))

    frac_real = float(n_real) / max(seq_len, 1)
    token_center = final_tokens.mean(dim=0, keepdim=True)
    token_spread = torch.norm(final_tokens - token_center, dim=1)

    layer_means: list[torch.Tensor] = []
    layer_norm_means: list[torch.Tensor] = []
    layer_norm_stds: list[torch.Tensor] = []
    for layer_id in layer_ids:
        layer_tokens = _real_token_states(hidden_states[layer_id], attention_mask)
        mean_vec = layer_tokens.mean(dim=0)
        norms = torch.norm(layer_tokens, dim=1)
        layer_means.append(mean_vec)
        layer_norm_means.append(norms.mean())
        layer_norm_stds.append(norms.std(unbiased=False))

    pairwise_cos: list[torch.Tensor] = []
    if len(layer_means) >= 2:
        for i in range(len(layer_means) - 1):
            pairwise_cos.append(
                F.cosine_similarity(layer_means[i], layer_means[i + 1], dim=0)
            )
    else:
        pairwise_cos.append(torch.tensor(1.0, dtype=hidden_states.dtype))

    edge_cos = F.cosine_similarity(layer_means[0], layer_means[-1], dim=0)

    geo = torch.tensor(
        [
            float(n_real),
            float(seq_len),
            frac_real,
            float(token_spread.mean()),
            float(token_spread.std(unbiased=False)),
            float(torch.stack(layer_norm_means).mean()),
            float(torch.stack(layer_norm_means).std(unbiased=False)),
            float(torch.stack(layer_norm_stds).mean()),
            float(torch.stack(layer_norm_stds).std(unbiased=False)),
            float(torch.stack(pairwise_cos).mean()),
            float(torch.stack(pairwise_cos).std(unbiased=False)),
            float(edge_cos),
        ],
        dtype=hidden_states.dtype,
    )
    return geo


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
