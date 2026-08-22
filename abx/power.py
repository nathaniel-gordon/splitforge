"""Power analysis: required sample size, minimum detectable effect, achieved power.

All formulas are the two-sided normal-approximation ones an experimentation team uses
for planning.  Two details that are usually got wrong and are handled correctly here:

* Sample size for a proportion uses the variance under the ALTERNATIVE
  (p_c(1-p_c) + p_t(1-p_t)) rather than 2*p_c(1-p_c); the difference matters once the
  MDE is large relative to the base rate.
* Unequal allocation is supported through `ratio = n_treatment / n_control`; a 90/10
  split needs far more total traffic than 50/50 for the same MDE, which is exactly
  the trade-off a ramp plan has to price in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class PowerPlan:
    """A sample-size / MDE plan for one metric."""

    metric: str
    kind: str
    base_value: float
    mde_rel: float
    mde_abs: float
    alpha: float
    power: float
    ratio: float
    n_control: int
    n_treatment: int
    n_total: int
    daily_traffic: float | None = None
    days_required: float | None = None
    variance_reduction: float = 0.0

    def table(self) -> str:
        from tabulate import tabulate

        rows = [
            ["metric", f"{self.metric} ({self.kind})"],
            ["baseline", f"{self.base_value:.6g}"],
            ["MDE (relative)", f"{self.mde_rel * 100:.2f}%"],
            ["MDE (absolute)", f"{self.mde_abs:.6g}"],
            ["alpha / power", f"{self.alpha:g} / {self.power:g}"],
            ["allocation (treat:control)", f"{self.ratio:g}:1"],
            ["variance reduction assumed", f"{self.variance_reduction * 100:.1f}%"],
            ["n per control arm", f"{self.n_control:,}"],
            ["n per treatment arm", f"{self.n_treatment:,}"],
            ["n total", f"{self.n_total:,}"],
        ]
        if self.daily_traffic:
            rows.append(["daily eligible traffic", f"{self.daily_traffic:,.0f}"])
            rows.append(["days required", f"{self.days_required:.1f}"])
        return tabulate(rows, headers=["planning input / output", "value"],
                        tablefmt="github")


def _z(alpha: float, power: float) -> tuple[float, float]:
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


def sample_size_proportion(base_rate: float, mde_rel: float, alpha: float = 0.05,
                           power: float = 0.80, ratio: float = 1.0,
                           variance_reduction: float = 0.0,
                           metric: str = "conversion",
                           daily_traffic: float | None = None) -> PowerPlan:
    """Per-arm sample size for detecting a relative lift `mde_rel` on a rate."""
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be in (0,1)")
    if mde_rel == 0:
        raise ValueError("mde_rel must be non-zero")
    p_t = base_rate * (1 + mde_rel)
    if not 0 < p_t < 1:
        raise ValueError("mde_rel pushes the treatment rate outside (0,1)")
    z_a, z_b = _z(alpha, power)
    var_c = base_rate * (1 - base_rate) * (1 - variance_reduction)
    var_t = p_t * (1 - p_t) * (1 - variance_reduction)
    delta = p_t - base_rate
    n_c = (z_a + z_b) ** 2 * (var_c + var_t / ratio) / delta**2
    n_c = int(math.ceil(n_c))
    n_t = int(math.ceil(n_c * ratio))
    total = n_c + n_t
    return PowerPlan(
        metric=metric, kind="proportion", base_value=base_rate, mde_rel=mde_rel,
        mde_abs=delta, alpha=alpha, power=power, ratio=ratio, n_control=n_c,
        n_treatment=n_t, n_total=total, daily_traffic=daily_traffic,
        days_required=(total / daily_traffic) if daily_traffic else None,
        variance_reduction=variance_reduction,
    )


def sample_size_continuous(base_mean: float, sd: float, mde_rel: float,
                           alpha: float = 0.05, power: float = 0.80, ratio: float = 1.0,
                           variance_reduction: float = 0.0,
                           metric: str = "revenue_per_user",
                           daily_traffic: float | None = None) -> PowerPlan:
    """Per-arm sample size for a continuous metric with standard deviation `sd`."""
    if sd <= 0:
        raise ValueError("sd must be positive")
    delta = base_mean * mde_rel
    if delta == 0:
        raise ValueError("mde_rel must be non-zero")
    z_a, z_b = _z(alpha, power)
    var = sd**2 * (1 - variance_reduction)
    n_c = (z_a + z_b) ** 2 * var * (1 + 1 / ratio) / delta**2
    n_c = int(math.ceil(n_c))
    n_t = int(math.ceil(n_c * ratio))
    total = n_c + n_t
    return PowerPlan(
        metric=metric, kind="continuous", base_value=base_mean, mde_rel=mde_rel,
        mde_abs=delta, alpha=alpha, power=power, ratio=ratio, n_control=n_c,
        n_treatment=n_t, n_total=total, daily_traffic=daily_traffic,
        days_required=(total / daily_traffic) if daily_traffic else None,
        variance_reduction=variance_reduction,
    )


def mde_proportion(base_rate: float, n_per_arm: int, alpha: float = 0.05,
                   power: float = 0.80, variance_reduction: float = 0.0) -> float:
    """Smallest RELATIVE lift detectable with `n_per_arm` users at the given power.

    Solved by root-finding on the exact sample-size formula rather than the usual
    2*p(1-p) shortcut, so it stays correct for large MDEs.
    """
    def gap(mde_rel: float) -> float:
        return sample_size_proportion(base_rate, mde_rel, alpha, power,
                                      variance_reduction=variance_reduction).n_control \
            - n_per_arm

    lo, hi = 1e-5, min(0.999 / base_rate - 1.0 - 1e-9, 5.0)
    if gap(hi) > 0:
        return float("inf")
    return float(optimize.brentq(gap, lo, hi, xtol=1e-8))


def achieved_power_proportion(base_rate: float, true_rel_lift: float, n_per_arm: int,
                              alpha: float = 0.05,
                              variance_reduction: float = 0.0) -> float:
    """Power of a fixed-horizon two-proportion test against a given true lift."""
    p_t = base_rate * (1 + true_rel_lift)
    delta = p_t - base_rate
    var = (base_rate * (1 - base_rate) + p_t * (1 - p_t)) * (1 - variance_reduction)
    se = math.sqrt(var / n_per_arm)
    z_a = stats.norm.ppf(1 - alpha / 2)
    lam = abs(delta) / se
    return float(stats.norm.sf(z_a - lam) + stats.norm.cdf(-z_a - lam))


def power_curve(base_rate: float, n_per_arm: int, alpha: float = 0.05,
                rel_lifts: np.ndarray | None = None,
                variance_reduction: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Power as a function of true relative lift, for plotting."""
    if rel_lifts is None:
        rel_lifts = np.linspace(0.0, 0.25, 60)
    powers = np.array([achieved_power_proportion(base_rate, r, n_per_arm, alpha,
                                                 variance_reduction)
                       for r in rel_lifts])
    return np.asarray(rel_lifts), powers
