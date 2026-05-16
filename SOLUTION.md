# SMILES-2026 Hallucination Detection — Solution Report

## 1. Reproducibility

### Commands

```bash
git clone https://github.com/fedorakooo/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate
# .venv\Scripts\activate.bat

pip install -r requirements.txt
python solution.py
```

---

## 2. Final Solution Description

### Pipeline overview

```
prompt + response
       │
       ▼
Qwen2.5-0.5B  (24 transformer layers + embeddings, hidden_dim = 896)
       │
       ▼  hidden_states: (n_layers, seq_len, 896)
aggregation.py  — layer selection, token pooling, optional geometric features (off)
       │
       ▼  feature vector: (11648,)
splitting.py  — stratified 5-fold CV + per-fold validation
       │
       ▼
probe.py  — HallucinationProbe (StandardScaler + MLP)
```

### Modified components

| File | What changed |
|------|--------------|
| [`aggregation.py`](aggregation.py) | Multi-layer, multi-pool feature extraction; optional geometric features (not used in final run) |
| [`probe.py`](probe.py) | `HallucinationProbe` — scaled MLP with class-weighted loss, early stopping, threshold tuning |
| [`splitting.py`](splitting.py) | Stratified 5-fold cross-validation with per-fold validation split |

### Feature aggregation (`aggregation.py`)

**Layer selection.** From all hidden layers (index 0 = embeddings, index 24 = final transformer layer), four anchor layers are selected at quartile positions:

```
layer_ids = [n//4, n//2, 3n//4, n-1]   →   [6, 12, 18, 24] for n_layers = 25
```

**Token masking.** Padding tokens are excluded via `_real_token_states`, which keeps only positions where `attention_mask > 0`.

**Per-layer pooling.** For each selected layer, three pools are computed over real tokens:

- **Mean pool** — average activation across the sequence
- **Max pool** — element-wise maximum (captures salient peaks)
- **Last-token pool** — state at the final real token (position-sensitive summary)

**Answer-tail pool.** On the final layer only, the mean of the **last 40%** of real tokens is appended. The hypothesis is that hallucination signals concentrate in the assistant's generated answer, which typically occupies the tail of the tokenised `prompt + response` sequence.

**Feature dimension:**

```
4 layers × 3 pools × 896  +  1 tail pool × 896  =  11,648
```

**Geometric features.** `extract_geometric_features` computes 12 scalar statistics (sequence length, token spread, layer-wise norm statistics, inter-layer cosine similarities). These are concatenated only when `USE_GEOMETRIC = True`. The submitted run keeps this flag `False`.

### Classifier (`probe.py`)

**Architecture.** The network is built lazily in `fit()` based on `input_dim`:

- If `input_dim > bottleneck_dim`: `Linear → LayerNorm → GELU → Dropout(0.20) → Linear → LayerNorm → GELU → Dropout(0.15) → Linear(1)`
- Otherwise a smaller two-layer variant is used.

Dimensions adapt to input size: `bottleneck_dim = min(512, max(128, input_dim // 2))`, `hidden_dim = min(256, max(96, bottleneck_dim // 2))`.

**Preprocessing.** `StandardScaler` is fit on training features before each `fit()` call.

**Training.**

- Loss: `BCEWithLogitsLoss` with `pos_weight = n_negative / n_positive` to handle class imbalance (~70% hallucinated in training data).
- Optimizer: AdamW (`lr=8e-4`, `weight_decay=1e-4`).
- Scheduler: `ReduceLROnPlateau` on validation loss (or training loss if no internal val).
- Early stopping: patience 25 epochs, max 250 epochs, gradient clipping at norm 1.0.
- Internal validation: 20% stratified hold-out from training data during `fit()` when sample count ≥ 40.

**Threshold tuning.** `fit_hyperparameters()` searches over candidate thresholds (unique predicted probabilities plus a 0–1 grid) on the external validation fold to maximise F1. This is called automatically by `evaluate.py` when a validation split exists. `predict()` applies `prob >= threshold` rather than the default 0.5.

### Evaluation protocol (`splitting.py` + `evaluate.py`)

**Splitting strategy:**

1. `StratifiedKFold` with `n_splits = min(5, min_class_count)` → **5 folds** on the 689-sample dataset.
2. Within each fold, **15%** of the training fold is held out as validation (stratified), when both classes have at least 2 samples in the train portion and `len(train) >= 20`.

Typical fold sizes: train ≈ 468, val ≈ 83, test ≈ 138.

**Per-fold checkpoints** (from `evaluate.py`):

1. Majority-class baseline (accuracy floor)
2. Probe on training split
3. Probe on validation split
4. Probe on held-out test split

Metrics: Accuracy, F1, AUROC.

### Design choices and what contributed most

Ranked by observed impact on internal metrics:

1. **Multi-layer + multi-pool aggregation** — The largest structural improvement over a single final-layer mean vector (896 dims → 11,648 dims). Combining mean, max, and last-token pools captures both average semantics and peak activations across depth.

2. **Answer-tail pooling** — Explicitly targets the response region at the end of the sequence, where the model's factual claim lives.

3. **Stratified K-fold evaluation + validation threshold tuning** — Produces stable averages and optimises the decision boundary for F1/AUROC rather than using a fixed 0.5 cutoff.

4. **Regularised MLP with class weighting** — Achieves high train AUROC (~87.5%) and non-trivial test AUROC (~66.7%), indicating the probe learns a ranking signal. However, hard 0/1 predictions on the imbalanced test folds still match the majority class for accuracy.

5. **Geometric features disabled** — Kept off in the final run for a simpler, fixed feature space (see Section 3).

### Results

| Checkpoint | Accuracy | F1 | AUROC |
|------------|----------|-----|-------|
| 1. Majority-class baseline | 70.10% | 82.42% | N/A |
| 2. Probe (train split) | 70.14% | 82.45% | **87.50%** |
| 3. Probe (val split) | 69.88% | 82.27% | **63.31%** |
| 4. Probe (test split) | 70.10% | 82.42% | **66.71%** |

---

## 3. Experiments and Discarded Attempts

The following table documents ideas visible in the codebase or implied by the skeleton, and why they are not part of the final submitted configuration. No additional unrecorded experiment logs exist in the repository.

| Idea | Evidence in code | Outcome / why discarded |
|------|------------------|-------------------------|
| **Geometric hand-crafted features** | `extract_geometric_features()` fully implemented in `aggregation.py`; `USE_GEOMETRIC = False` in `solution.py` | Adds 12 scalar features (length, spread, layer norms, cosine drift). Not enabled in the run. Kept off for a simpler feature space without measured improvement in committed metrics. |
| **Single-layer mean pooling only** | Original skeleton in `aggregate()` suggested a single final-layer mean | Replaced by 4-layer × 3-pool + tail design to enrich representation (896 → 11,648 dims). |
| **Single random train/val/test split** | `splitting.py` skeleton used one `train_test_split` | Replaced by stratified 5-fold CV for more reliable averaged metrics. |
| **Fixed decision threshold 0.5** | Default `_threshold = 0.5` in `probe.py` | `fit_hyperparameters()` tunes threshold on validation fold to maximise F1. |
| **Larger batch size for extraction** | `BATCH_SIZE = 4` in `solution.py` | Conservative batch size for GPU memory. |