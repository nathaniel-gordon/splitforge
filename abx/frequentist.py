"""Frequentist two-sample tests: two-proportion z-test and Welch's t-test.

Design notes (the WHY):
* The z-test statistic uses the POOLED variance (correct under H0, which is what a
  p-value conditions on) while the confidence interval uses the UNPOOLED variance
  (correct under the estimated alternative).  Mixing these up is the single most
  common bug in home-grown experiment tooling and produces CIs that disagree with
  their own p-value near the boundary.
* Relative lift intervals use the delta method on log(p_t / p_c) rather than
  naively dividing the absolute interval by the control estimate; the log scale
  keeps the interval asymmetric and strictly positive, which is how ratios behave.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Outcome of one two-sample comparison on one metric."""

    metric: str
    kind: str  # "proportion" | "continuous"
    n_control: int
    n_treatment: int
    est_control: float
    est_treatment: float
    abs_lift: float
    abs_ci: tuple[float, float]
    rel_lift: float
    rel_ci: tuple[float, float]
    se_abs: float
    statistic: float
    p_value: float
    dof: float = float("nan")
    alpha: float = 0.05
    extra: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def _log_ratio_ci(est_c: float, est_t: float, se_c: float, se_t: float,
                  z: float) -> tuple[float, float]:
    """Delta-method CI for the RELATIVE lift via the log of the ratio.

    Returns (low, high) on the relative-lift scale (0.05 == +5%).  Falls back to
    a symmetric interval when either estimate is non-positive (log undefined).
    """
    if est_c <= 0 or est_t <= 0:
        se_ratio = math.sqrt(se_t**2 + (est_t / est_c) ** 2 * se_c**2) / abs(est_c) \
            if est_c != 0 else float("nan")
        centre = est_t / est_c - 1.0 if est_c != 0 else float("nan")
        return (centre - z * se_ratio, centre + z * se_ratio)
    se_log = math.sqrt((se_t / est_t) ** 2 + (se_c / est_c) ** 2)
    centre_log = math.log(est_t / est_c)
    return (math.exp(centre_log - z * se_log) - 1.0,
            math.exp(centre_log + z * se_log) - 1.0)


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int,
                         alpha: float = 0.05, metric: str = "conversion") -> TestResult:
    """Two-sided two-proportion z-test with pooled test SE and unpooled CI SE."""
    if n_c <= 0 or n_t <= 0:
        raise ValueError("both arms need at least one unit")
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t), 1e-300))
    se_unpool = math.sqrt(max(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t, 1e-300))
    diff = p_t - p_c
    z_stat = diff / se_pool
    p_value = 2.0 * stats.norm.sf(abs(z_stat))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    abs_ci = (diff - z_crit * se_unpool, diff + z_crit * se_unpool)
    rel_ci = _log_ratio_ci(p_c, p_t,
                           math.sqrt(p_c * (1 - p_c) / n_c),
                           math.sqrt(p_t * (1 - p_t) / n_t), z_crit)
    return TestResult(
        metric=metric, kind="proportion", n_control=n_c, n_treatment=n_t,
        est_control=p_c, est_treatment=p_t, abs_lift=diff, abs_ci=abs_ci,
        rel_lift=(p_t / p_c - 1.0) if p_c > 0 else float("nan"), rel_ci=rel_ci,
        se_abs=se_unpool, statistic=z_stat, p_value=p_value, alpha=alpha,
        extra={"x_control": int(x_c), "x_treatment": int(x_t), "se_pooled": se_pool},
    )


def welch_ttest(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05,
                metric: str = "revenue_per_user") -> TestResult:
    """Welch's unequal-variance t-test.

    Welch (not Student) because A/B arms routinely have different variances once a
    treatment changes behaviour; the pooled-variance t-test is anti-conservative
    exactly when the treatment is doing something interesting.
    """
    y_c = np.asarray(y_c, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    n_c, n_t = y_c.size, y_t.size
    if n_c < 2 or n_t < 2:
        raise ValueError("Welch t-test needs at least 2 observations per arm")
    m_c, m_t = float(y_c.mean()), float(y_t.mean())
    v_c = float(y_c.var(ddof=1))
    v_t = float(y_t.var(ddof=1))
    se_c, se_t = math.sqrt(v_c / n_c), math.sqrt(v_t / n_t)
    se = math.sqrt(se_c**2 + se_t**2)
    diff = m_t - m_c
    t_stat = diff / se if se > 0 else 0.0
    # Welch-Satterthwaite effective degrees of freedom.
    dof = (se_c**2 + se_t**2) ** 2 / (
        se_c**4 / (n_c - 1) + se_t**4 / (n_t - 1))
    p_value = 2.0 * stats.t.sf(abs(t_stat), dof)
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    abs_ci = (diff - t_crit * se, diff + t_crit * se)
    rel_ci = _log_ratio_ci(m_c, m_t, se_c, se_t, t_crit)
    return TestResult(
        metric=metric, kind="continuous", n_control=n_c, n_treatment=n_t,
        est_control=m_c, est_treatment=m_t, abs_lift=diff, abs_ci=abs_ci,
        rel_lift=(m_t / m_c - 1.0) if m_c != 0 else float("nan"), rel_ci=rel_ci,
        se_abs=se, statistic=t_stat, p_value=p_value, dof=dof, alpha=alpha,
        extra={"sd_control": math.sqrt(v_c), "sd_treatment": math.sqrt(v_t)},
    )


def format_results(results: list[TestResult]) -> str:
    """Render a list of TestResult as an aligned table."""
    from tabulate import tabulate

    headers = ["metric", "control", "treatment", "abs lift", "abs 95% CI",
               "rel lift", "rel 95% CI", "stat", "p", "sig"]
    rows = []
    for r in results:
        rows.append([
            r.metric, f"{r.est_control:.5g}", f"{r.est_treatment:.5g}",
            f"{r.abs_lift:+.4g}",
            f"[{r.abs_ci[0]:+.4g}, {r.abs_ci[1]:+.4g}]",
            f"{r.rel_lift * 100:+.2f}%",
            f"[{r.rel_ci[0] * 100:+.2f}%, {r.rel_ci[1] * 100:+.2f}%]",
            f"{r.statistic:+.3f}", f"{r.p_value:.3g}",
            "YES" if r.significant else "no",
        ])
    return tabulate(rows, headers=headers, tablefmt="github")
