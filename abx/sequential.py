"""Sequential / always-valid inference, plus a demonstration of why you need it.

Three procedures live here:

1. `naive_peek_decision` - reject as soon as any interim |z| > z_{alpha/2}.  This is
   what a dashboard with a "p-value" tile does, and it is WRONG: under the null the
   test statistic is a Brownian motion, so with K looks the chance that it ever
   wanders past a fixed boundary grows roughly like sqrt(K).

2. `alpha_spending_boundaries` - Lan-DeMets alpha spending with the O'Brien-Fleming
   spending function, boundaries obtained by the Armitage-McPherson-Rowe recursion:
   propagate the sub-density of the Brownian B-value through each interim, truncated
   to the continuation region, and solve for the next boundary so that the newly
   spent error equals the increment of the spending function.  Exact up to the
   quadrature grid - no simulation, no lookup tables.

3. `msprt_pvalue` - a mixture Sequential Probability Ratio Test.  Mixing the
   alternative over theta ~ N(0, tau^2) makes the likelihood ratio a non-negative
   martingale under the null, so Ville's inequality gives
   P(sup_n Lambda_n >= 1/alpha) <= alpha.  The running minimum of 1/Lambda is
   therefore an ALWAYS-VALID p-value: you may look at it continuously, at any
   arbitrary times, and stop whenever you like.

The B-value parameterisation: with information fraction t = n/n_max, the statistic
B(t) = Z(t) * sqrt(t) is (asymptotically) standard Brownian motion under H0, so its
increments are independent N(0, dt).  Independence is what makes the recursion below
a simple convolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Alpha-spending boundaries
# --------------------------------------------------------------------------- #

def obrien_fleming_spend(t: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Lan-DeMets O'Brien-Fleming spending function, two-sided form.

        alpha*(t) = 4 * (1 - Phi(z_{alpha/4} / sqrt(t)))

    The alpha/4 (rather than alpha/2) is the two-sided version: apply the one-sided
    spending function at level alpha/2 to each tail and add them.  It reproduces the
    published Lan-DeMets boundary tables exactly, and it spends almost nothing early -
    which is the whole point of OBF, since stopping at 10% information should require
    overwhelming evidence.
    """
    t = np.clip(np.asarray(t, dtype=float), 1e-12, 1.0)
    z_a = stats.norm.ppf(1 - alpha / 4)
    return np.minimum(4.0 * stats.norm.sf(z_a / np.sqrt(t)), alpha)


def pocock_spend(t: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Lan-DeMets Pocock spending function: alpha * log(1 + (e-1) t)."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return alpha * np.log1p((math.e - 1.0) * t)


SPENDING = {"obf": obrien_fleming_spend, "pocock": pocock_spend}


@dataclass
class Boundaries:
    """Two-sided group-sequential boundaries on both the B and Z scales."""

    info_fractions: np.ndarray
    z_boundaries: np.ndarray
    b_boundaries: np.ndarray
    alpha_spent: np.ndarray       # cumulative
    alpha_increment: np.ndarray   # per look
    spending: str
    alpha: float


def alpha_spending_boundaries(info_fractions: np.ndarray | list[float], alpha: float = 0.05,
                              spending: str = "obf", n_grid: int = 1201,
                              span: float = 7.0) -> Boundaries:
    """Group-sequential boundaries by numerical propagation of the B-value density.

    `info_fractions` must be increasing in (0, 1].  Returns boundaries such that the
    overall two-sided type-I error of "stop the first time |Z_k| >= z_k" is `alpha`.
    """
    t = np.asarray(info_fractions, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("info_fractions must be a non-empty 1-D sequence")
    if np.any(np.diff(t) <= 0) or t[0] <= 0 or t[-1] > 1.0 + 1e-12:
        raise ValueError("info_fractions must be strictly increasing within (0, 1]")
    if spending not in SPENDING:
        raise ValueError(f"unknown spending function {spending!r}")

    cum = np.asarray(SPENDING[spending](t, alpha), dtype=float)
    inc = np.diff(np.concatenate([[0.0], cum]))
    inc = np.maximum(inc, 0.0)

    x = np.linspace(-span, span, n_grid)
    dx = x[1] - x[0]
    b_bounds = np.empty(t.size)

    # Stage 1: B(t1) ~ N(0, t1) exactly, so the boundary is closed-form.
    dens = stats.norm.pdf(x, loc=0.0, scale=math.sqrt(t[0]))
    b_bounds[0] = math.sqrt(t[0]) * stats.norm.ppf(1 - inc[0] / 2) if inc[0] > 0 else span
    b_bounds[0] = min(b_bounds[0], span)
    dens = np.where(np.abs(x) >= b_bounds[0], 0.0, dens)

    for k in range(1, t.size):
        s = math.sqrt(t[k] - t[k - 1])
        # Sub-density of B(t_k) restricted to "no crossing yet": convolution with the
        # independent Gaussian increment.
        kernel = stats.norm.pdf((x[:, None] - x[None, :]) / s) / s
        dens = kernel @ (dens * dx)
        b_bounds[k] = _solve_boundary(x, dens, dx, inc[k], span)
        dens = np.where(np.abs(x) >= b_bounds[k], 0.0, dens)

    z_bounds = b_bounds / np.sqrt(t)
    return Boundaries(info_fractions=t, z_boundaries=z_bounds, b_boundaries=b_bounds,
                      alpha_spent=cum, alpha_increment=inc, spending=spending, alpha=alpha)


def _solve_boundary(x: np.ndarray, dens: np.ndarray, dx: float, target: float,
                    span: float) -> float:
    """Smallest b with P(|B| >= b, no earlier crossing) == target, by bisection."""
    if target <= 0:
        return span

    def tail(b: float) -> float:
        mask = np.abs(x) >= b
        return float(np.sum(dens[mask]) * dx)

    if tail(0.0) <= target:  # not enough surviving mass to spend the budget
        return 0.0
    lo, hi = 0.0, span
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tail(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Mixture SPRT (always-valid p-values)
# --------------------------------------------------------------------------- #

def msprt_likelihood_ratio(z: np.ndarray, se: np.ndarray, tau: float) -> np.ndarray:
    """Mixture likelihood ratio for a normal-mean test, mixing theta ~ N(0, tau^2).

    With v = se^2 the sampling variance of the effect estimate,
        Lambda = sqrt(v / (v + tau^2)) * exp( z^2 / 2 * tau^2 / (v + tau^2) ).
    """
    z = np.asarray(z, dtype=float)
    v = np.asarray(se, dtype=float) ** 2
    tau2 = float(tau) ** 2
    if tau2 <= 0:
        raise ValueError("tau (the mixing scale) must be positive")
    shrink = tau2 / (v + tau2)
    log_lambda = 0.5 * np.log(v / (v + tau2)) + 0.5 * z**2 * shrink
    return np.exp(log_lambda)


def always_valid_pvalues(z: np.ndarray, se: np.ndarray, tau: float) -> np.ndarray:
    """Running-minimum always-valid p-value sequence from the mixture SPRT."""
    lam = msprt_likelihood_ratio(z, se, tau)
    with np.errstate(divide="ignore"):
        p = np.minimum(1.0, 1.0 / lam)
    return np.minimum.accumulate(p, axis=-1)


# --------------------------------------------------------------------------- #
# Peeking simulation
# --------------------------------------------------------------------------- #

@dataclass
class PeekingSimulation:
    """False-positive (or power) rates of three monitoring policies."""

    n_sims: int
    n_looks: int
    n_per_arm_final: int
    base_rate: float
    true_rel_lift: float
    alpha: float
    tau: float
    naive_rate: float
    obf_rate: float
    msprt_rate: float
    fixed_horizon_rate: float
    boundaries: Boundaries
    z_paths: np.ndarray            # (n_sims, n_looks)
    av_p_paths: np.ndarray         # (n_sims, n_looks) always-valid p-values
    naive_hit: np.ndarray
    obf_hit: np.ndarray
    msprt_hit: np.ndarray

    def example_false_positive(self) -> int | None:
        """Index of a simulated run the naive rule calls but no valid rule does.

        Printing one concrete trajectory is far more convincing than a rate: it shows
        a specific A/A test whose dashboard went green on some middle day.
        """
        idx = np.flatnonzero(self.naive_hit & ~self.obf_hit & ~self.msprt_hit)
        return int(idx[0]) if idx.size else None

    @property
    def is_null(self) -> bool:
        return abs(self.true_rel_lift) < 1e-12

    @property
    def rate_label(self) -> str:
        return "false-positive rate" if self.is_null else "power"

    def as_rows(self) -> list[list[str]]:
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        return [
            ["fixed horizon (single look, correct by construction)",
             f"{self.fixed_horizon_rate:.4f}", "reference"],
            [f"naive peeking ({self.n_looks} looks, |z|>{z_crit:.2f})",
             f"{self.naive_rate:.4f}",
             "INFLATED" if self.is_null and self.naive_rate > 1.5 * self.alpha else "-"],
            [f"O'Brien-Fleming alpha spending ({self.n_looks} looks)",
             f"{self.obf_rate:.4f}", "controlled" if self.is_null else "-"],
            [f"mixture SPRT / always-valid ({self.n_looks} looks)",
             f"{self.msprt_rate:.4f}", "controlled" if self.is_null else "-"],
        ]

    def table(self) -> str:
        from tabulate import tabulate

        return tabulate(self.as_rows(),
                        headers=["monitoring policy", self.rate_label, "verdict"],
                        tablefmt="github")


def peeking_simulation(n_sims: int = 2000, n_looks: int = 10,
                       n_per_arm_final: int = 20000, base_rate: float = 0.10,
                       true_rel_lift: float = 0.0, alpha: float = 0.05,
                       tau: float | None = None, seed: int = 42,
                       boundaries: Boundaries | None = None) -> PeekingSimulation:
    """Simulate repeated interim looks and measure each policy's rejection rate.

    Under `true_rel_lift == 0` the rejection rate IS the false-positive rate, and the
    naive policy will visibly blow past `alpha` while the two sequential policies
    stay at or below it.
    """
    rng = np.random.default_rng(seed)
    looks = np.round(np.linspace(n_per_arm_final / n_looks, n_per_arm_final,
                                 n_looks)).astype(int)
    increments = np.diff(np.concatenate([[0], looks]))
    p_c = base_rate
    p_t = base_rate * (1.0 + true_rel_lift)
    if not 0.0 < p_t < 1.0:
        raise ValueError("true_rel_lift moves the treatment rate outside (0,1)")

    x_c = np.zeros(n_sims, dtype=np.int64)
    x_t = np.zeros(n_sims, dtype=np.int64)
    z = np.empty((n_sims, n_looks))
    se = np.empty((n_sims, n_looks))
    for k, inc in enumerate(increments):
        x_c += rng.binomial(inc, p_c, size=n_sims)
        x_t += rng.binomial(inc, p_t, size=n_sims)
        n_k = looks[k]
        phat_c = x_c / n_k
        phat_t = x_t / n_k
        pool = (x_c + x_t) / (2 * n_k)
        se_k = np.sqrt(np.maximum(pool * (1 - pool) * (2.0 / n_k), 1e-300))
        z[:, k] = (phat_t - phat_c) / se_k
        se[:, k] = se_k

    if tau is None:
        # Default mixing scale: a 10% relative change in the base rate - the size of
        # effect the experiment is actually powered to care about.
        tau = 0.10 * base_rate
    if boundaries is None:
        boundaries = alpha_spending_boundaries(looks / looks[-1], alpha=alpha,
                                               spending="obf")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    naive = np.any(np.abs(z) >= z_crit, axis=1)
    obf = np.any(np.abs(z) >= boundaries.z_boundaries[None, :], axis=1)
    av_p = always_valid_pvalues(z, se, tau)
    msprt = np.any(av_p <= alpha, axis=1)
    fixed = np.abs(z[:, -1]) >= z_crit

    return PeekingSimulation(
        n_sims=n_sims, n_looks=n_looks, n_per_arm_final=n_per_arm_final,
        base_rate=base_rate, true_rel_lift=true_rel_lift, alpha=alpha, tau=float(tau),
        naive_rate=float(naive.mean()), obf_rate=float(obf.mean()),
        msprt_rate=float(msprt.mean()), fixed_horizon_rate=float(fixed.mean()),
        boundaries=boundaries, z_paths=z, av_p_paths=av_p,
        naive_hit=naive, obf_hit=obf, msprt_hit=msprt,
    )


# --------------------------------------------------------------------------- #
# Monitoring a real experiment
# --------------------------------------------------------------------------- #

@dataclass
class SequentialMonitor:
    """Interim trajectory of one real experiment under all three policies."""

    looks: np.ndarray            # cumulative units across BOTH arms at each look
    info_fractions: np.ndarray
    z: np.ndarray
    se: np.ndarray
    always_valid_p: np.ndarray
    naive_p: np.ndarray
    boundaries: Boundaries
    tau: float
    alpha: float

    @property
    def naive_first_cross(self) -> int | None:
        idx = np.flatnonzero(self.naive_p <= self.alpha)
        return int(idx[0]) if idx.size else None

    @property
    def obf_first_cross(self) -> int | None:
        idx = np.flatnonzero(np.abs(self.z) >= self.boundaries.z_boundaries)
        return int(idx[0]) if idx.size else None

    @property
    def msprt_first_cross(self) -> int | None:
        idx = np.flatnonzero(self.always_valid_p <= self.alpha)
        return int(idx[0]) if idx.size else None

    @property
    def stopped_for_significance(self) -> bool:
        return self.msprt_first_cross is not None or self.obf_first_cross is not None

    def table(self) -> str:
        from tabulate import tabulate

        rows = []
        for k in range(self.looks.size):
            rows.append([
                k + 1, int(self.looks[k]), f"{self.info_fractions[k]:.2f}",
                f"{self.z[k]:+.3f}", f"{self.naive_p[k]:.4f}",
                f"{self.boundaries.z_boundaries[k]:.3f}",
                "cross" if abs(self.z[k]) >= self.boundaries.z_boundaries[k] else "",
                f"{self.always_valid_p[k]:.4f}",
                "cross" if self.always_valid_p[k] <= self.alpha else "",
            ])
        return tabulate(rows, headers=["look", "n total", "info", "z", "naive p",
                                       "OBF z-bound", "OBF", "always-valid p", "mSPRT"],
                        tablefmt="github")


def monitor_experiment(x_c: np.ndarray, n_c: np.ndarray, x_t: np.ndarray,
                       n_t: np.ndarray, alpha: float = 0.05,
                       tau: float | None = None,
                       spending: str = "obf") -> SequentialMonitor:
    """Run all three monitoring policies over an experiment's cumulative interim data."""
    x_c = np.asarray(x_c, dtype=float)
    n_c = np.asarray(n_c, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    n_t = np.asarray(n_t, dtype=float)
    if not (x_c.shape == n_c.shape == x_t.shape == n_t.shape):
        raise ValueError("interim arrays must have the same shape")
    p_c, p_t = x_c / n_c, x_t / n_t
    pool = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(np.maximum(pool * (1 - pool) * (1 / n_c + 1 / n_t), 1e-300))
    z = (p_t - p_c) / se
    naive_p = 2.0 * stats.norm.sf(np.abs(z))
    info = (n_c + n_t) / (n_c[-1] + n_t[-1])
    if tau is None:
        tau = 0.10 * float(pool[-1])
    bounds = alpha_spending_boundaries(info, alpha=alpha, spending=spending)
    av_p = always_valid_pvalues(z, se, tau)
    return SequentialMonitor(looks=n_c + n_t, info_fractions=info, z=z, se=se,
                             always_valid_p=av_p, naive_p=naive_p, boundaries=bounds,
                             tau=float(tau), alpha=alpha)
