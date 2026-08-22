"""CUPED - Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013).

Replace the metric Y by  Y_adj = Y - theta * (X - E[X])  where X is a PRE-period
covariate.  Because X is measured before assignment, E[X] is identical in both arms
in expectation, so the adjustment is unbiased for the treatment effect: subtracting
it removes pre-existing user-level variance without touching the estimand.

theta = Cov(Y, X) / Var(X) minimises Var(Y_adj), giving
    Var(Y_adj) = Var(Y) * (1 - rho^2)
so the variance reduction equals the squared correlation, and the effective sample
size is multiplied by 1 / (1 - rho^2).  A 30% variance reduction is worth ~43% more
users - i.e. it is usually the single cheapest source of experimental power.

Critical constraint enforced here: theta is estimated on the POOLED data across arms.
Estimating it per arm would let the treatment influence theta and reintroduce bias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .frequentist import TestResult, welch_ttest


@dataclass
class CUPEDResult:
    """Before/after comparison of a metric with and without CUPED adjustment."""

    metric: str
    covariate: str
    theta: float
    correlation: float
    var_raw: float
    var_adjusted: float
    variance_reduction: float      # 1 - var_adj/var_raw
    ess_multiplier: float          # effective sample-size gain
    se_raw: float
    se_adjusted: float
    raw_test: TestResult
    adjusted_test: TestResult
    balance_p_value: float         # pre-period covariate balance across arms

    def table(self) -> str:
        from tabulate import tabulate

        rows = [
            ["theta = Cov(Y,X)/Var(X)", f"{self.theta:.6g}"],
            ["corr(Y, X)", f"{self.correlation:.4f}"],
            ["Var(Y) raw", f"{self.var_raw:.6g}"],
            ["Var(Y) after CUPED", f"{self.var_adjusted:.6g}"],
            ["variance reduction", f"{self.variance_reduction * 100:.2f}%"],
            ["effective sample-size multiplier", f"{self.ess_multiplier:.3f}x"],
            ["SE of lift, raw", f"{self.se_raw:.6g}"],
            ["SE of lift, CUPED", f"{self.se_adjusted:.6g}"],
            ["SE shrink factor", f"{self.se_adjusted / self.se_raw:.4f}"],
            ["p-value raw -> CUPED",
             f"{self.raw_test.p_value:.4g} -> {self.adjusted_test.p_value:.4g}"],
            ["pre-period covariate balance p",
             f"{self.balance_p_value:.4f} (>0.01 expected: X must not differ by arm)"],
        ]
        return tabulate(rows, headers=["CUPED diagnostic", "value"], tablefmt="github")


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (adjusted y, theta, correlation) using pooled theta."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError("metric and covariate must have the same length")
    var_x = float(x.var(ddof=1))
    if var_x <= 0:
        return y.copy(), 0.0, 0.0
    cov = float(np.cov(y, x, ddof=1)[0, 1])
    theta = cov / var_x
    sd_y = float(y.std(ddof=1))
    rho = cov / (sd_y * math.sqrt(var_x)) if sd_y > 0 else 0.0
    return y - theta * (x - float(x.mean())), theta, rho


def apply_cuped(y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray,
                *, metric: str = "revenue_per_user", covariate: str = "pre_revenue",
                alpha: float = 0.05) -> CUPEDResult:
    """Fit pooled theta, adjust both arms, and re-run the Welch test."""
    y_c = np.asarray(y_c, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    x_c = np.asarray(x_c, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])
    adj_all, theta, rho = cuped_adjust(y_all, x_all)
    adj_c, adj_t = adj_all[: y_c.size], adj_all[y_c.size:]

    raw = welch_ttest(y_c, y_t, alpha=alpha, metric=metric)
    adjusted = welch_ttest(adj_c, adj_t, alpha=alpha, metric=f"{metric} (CUPED)")
    balance = welch_ttest(x_c, x_t, alpha=alpha, metric=covariate)

    var_raw = float(y_all.var(ddof=1))
    var_adj = float(adj_all.var(ddof=1))
    reduction = 1.0 - var_adj / var_raw if var_raw > 0 else 0.0
    return CUPEDResult(
        metric=metric, covariate=covariate, theta=theta, correlation=rho,
        var_raw=var_raw, var_adjusted=var_adj, variance_reduction=reduction,
        ess_multiplier=1.0 / max(1.0 - reduction, 1e-12),
        se_raw=raw.se_abs, se_adjusted=adjusted.se_abs,
        raw_test=raw, adjusted_test=adjusted, balance_p_value=balance.p_value,
    )
