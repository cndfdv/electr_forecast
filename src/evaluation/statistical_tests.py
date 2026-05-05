"""Statistical tests for forecast comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float | str | bool]:
    """Paired t-test on per-window losses."""
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    stat, p_value = stats.ttest_rel(errors_a, errors_b, nan_policy="omit")
    return {
        "test_name": "paired_t_test",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float | str | bool]:
    """Simple Diebold-Mariano test with no small-sample correction."""
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    d = errors_a - errors_b
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"test_name": "diebold_mariano", "statistic": float("nan"), "p_value": float("nan"), "significant": False}
    dm_stat = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {
        "test_name": "diebold_mariano",
        "statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }

