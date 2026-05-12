"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._seed: int = 42
        self._batch_size: int = 64
        self._max_epochs: int = 250
        self._patience: int = 25

    def _set_seed(self) -> None:
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        bottleneck_dim = min(512, max(128, input_dim // 2))
        hidden_dim = min(256, max(96, bottleneck_dim // 2))

        if input_dim > bottleneck_dim:
            self._net = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self._net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, 1),
            )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        self._set_seed()
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)
        y_float = y.astype(np.float32)

        self._build_network(X_scaled.shape[1])

        idx_all = np.arange(len(y))
        can_stratify = len(np.unique(y)) > 1 and np.min(np.bincount(y.astype(int))) >= 2
        use_internal_val = len(y) >= 40 and can_stratify

        if use_internal_val:
            idx_train, idx_val = train_test_split(
                idx_all,
                test_size=0.2,
                random_state=self._seed,
                stratify=y,
            )
        else:
            idx_train, idx_val = idx_all, None

        X_train = torch.from_numpy(X_scaled[idx_train]).float()
        y_train = torch.from_numpy(y_float[idx_train])

        X_val = None
        y_val = None
        if idx_val is not None:
            X_val = torch.from_numpy(X_scaled[idx_val]).float()
            y_val = torch.from_numpy(y_float[idx_val])

        n_pos = int(y_train.sum().item())
        n_neg = int(y_train.numel() - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self.parameters(), lr=8e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
        )

        best_score = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        bad_epochs = 0
        min_delta = 1e-4

        for _ in range(self._max_epochs):
            self.train()
            perm = torch.randperm(X_train.size(0))
            epoch_loss = 0.0

            for start in range(0, X_train.size(0), self._batch_size):
                batch_idx = perm[start : start + self._batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.item()) * xb.size(0)

            train_loss = epoch_loss / max(X_train.size(0), 1)

            self.eval()
            with torch.no_grad():
                if X_val is not None and y_val is not None:
                    val_logits = self(X_val)
                    val_loss = float(criterion(val_logits, y_val).item())
                    score = val_loss
                else:
                    score = train_loss

            scheduler.step(score)

            if score < best_score - min_delta:
                best_score = score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self._patience:
                break

        self.load_state_dict(best_state)
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

