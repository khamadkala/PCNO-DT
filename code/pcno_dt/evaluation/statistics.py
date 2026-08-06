from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedComparison:
    mean_difference: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    cohen_d: float
    corrected_alpha: float
    significant: bool


def student_interval(values: NDArray[np.float64], confidence: float = 0.95) -> ConfidenceInterval:
    values = np.asarray(values, dtype=np.float64)
    estimate = float(values.mean())
    if values.size < 2:
        return ConfidenceInterval(estimate, float("nan"), float("nan"))
    sem = stats.sem(values)
    critical = stats.t.ppf((1.0 + confidence) / 2.0, values.size - 1)
    return ConfidenceInterval(estimate, estimate - critical * sem, estimate + critical * sem)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    z = stats.norm.ppf((1.0 + confidence) / 2.0)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
    return ConfidenceInterval(proportion, center - half / denominator, center + half / denominator)


def bootstrap_interval(
    values: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap values cannot be empty")
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        resampled = generator.choice(values, size=values.size, replace=True)
        estimates[index] = statistic(resampled)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, (alpha, 1.0 - alpha))
    return ConfidenceInterval(float(statistic(values)), float(lower), float(upper))


def paired_comparison(
    proposed: NDArray[np.float64],
    baseline: NDArray[np.float64],
    comparisons: int = 13,
    alpha: float = 0.05,
) -> PairedComparison:
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if proposed.shape != baseline.shape:
        raise ValueError("paired arrays must have identical shapes")
    differences = proposed - baseline
    t_result = stats.ttest_rel(proposed, baseline)
    w_result = stats.wilcoxon(proposed, baseline)
    standard_deviation = differences.std(ddof=1)
    effect = differences.mean() / standard_deviation if standard_deviation > 0 else float("inf")
    corrected = alpha / comparisons
    return PairedComparison(
        mean_difference=float(differences.mean()),
        t_statistic=float(t_result.statistic),
        t_pvalue=float(t_result.pvalue),
        wilcoxon_statistic=float(w_result.statistic),
        wilcoxon_pvalue=float(w_result.pvalue),
        cohen_d=float(effect),
        corrected_alpha=corrected,
        significant=bool(t_result.pvalue < corrected and w_result.pvalue < corrected),
    )


def bland_altman(reference: NDArray[np.float64], estimate: NDArray[np.float64]) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    difference = estimate - reference
    mean = (estimate + reference) / 2.0
    bias = float(difference.mean())
    spread = float(difference.std(ddof=1))
    slope, intercept, r_value, p_value, _ = stats.linregress(mean, difference)
    return {
        "bias": bias,
        "lower_agreement": bias - 1.96 * spread,
        "upper_agreement": bias + 1.96 * spread,
        "proportional_bias_slope": float(slope),
        "proportional_bias_intercept": float(intercept),
        "proportional_bias_r": float(r_value),
        "proportional_bias_p": float(p_value),
    }

