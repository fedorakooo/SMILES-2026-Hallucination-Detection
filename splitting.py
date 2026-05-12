"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _validate_split(
    idx_train: np.ndarray,
    idx_val: np.ndarray | None,
    idx_test: np.ndarray,
    n_samples: int,
) -> None:
    parts = [idx_train, idx_test] if idx_val is None else [idx_train, idx_val, idx_test]
    sets = [set(map(int, p.tolist())) for p in parts]

    union = set().union(*sets)
    if union != set(range(n_samples)):
        raise ValueError("Split indices must cover all rows exactly once.")

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i].intersection(sets[j]):
                raise ValueError("Split indices must be non-overlapping.")


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs a single stratified random split preserving
    the class ratio in each subset.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    n_samples = len(y)
    idx = np.arange(n_samples, dtype=int)
    y = np.asarray(y).astype(int)

    if n_samples < 10 or len(np.unique(y)) < 2:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=min(max(test_size, 0.1), 0.5),
            random_state=random_state,
            shuffle=True,
        )
        idx_train, idx_test = np.asarray(idx_train, dtype=int), np.asarray(idx_test, dtype=int)
        idx_val = None
        _validate_split(idx_train, idx_val, idx_test, n_samples)
        return [(idx_train, idx_val, idx_test)]

    class_counts = np.bincount(y)
    min_class = int(class_counts.min()) if class_counts.size > 1 else 1
    n_splits = max(2, min(5, min_class))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for idx_train_full, idx_test in skf.split(idx, y):
        idx_train_full = idx[idx_train_full]
        idx_test = idx[idx_test]

        val_fraction = min(max(val_size, 0.05), 0.35)
        can_make_val = (
            len(np.unique(y[idx_train_full])) > 1
            and np.min(np.bincount(y[idx_train_full])) >= 2
            and len(idx_train_full) >= 20
        )

        if can_make_val:
            idx_train, idx_val = train_test_split(
                idx_train_full,
                test_size=val_fraction,
                random_state=random_state,
                stratify=y[idx_train_full],
            )
        else:
            idx_train, idx_val = idx_train_full, None

        idx_train = np.asarray(idx_train, dtype=int)
        idx_test = np.asarray(idx_test, dtype=int)
        idx_val = None if idx_val is None else np.asarray(idx_val, dtype=int)

        _validate_split(idx_train, idx_val, idx_test, n_samples)
        splits.append((idx_train, idx_val, idx_test))

    return splits

