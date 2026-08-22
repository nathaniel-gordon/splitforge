"""Bayesian A/B analysis: Beta-Binomial for rates, Normal-Normal for continuous.

Everything here is computed by deterministic numerical integration, never Monte
Carlo.  WHY: a decision engine that returns a slightly different "P(variant wins)"
on every run is impossible to audit, impossible to unit-test at a real threshold,
and erodes trust with stakeholders who re-run the same report.

Two quantities drive the decision rule:
  * P(treatment > control) - the posterior probability of a win.
  * Expected loss - E[max(control - treatment, 0)], the average amount of metric
    you give up if you ship the treatment and it is in fact worse.  Expected loss
    is the right stopping rule (Bayesian risk), because a 92% win probability with
    a negligible downside is a ship, while a 96% win probability with a fat left
    tail is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats


@dataclass
class BayesResult:
    """Posterior summary for one metric."""

    metric: str
    kind: str
    prior: str
    post_mean_control: float
    post_mean_treatment: float
    prob_treatment_better: float
    expected_loss_ship: float      # loss (metric units) from shipping treatment
    expected_loss_keep: float      # loss from keeping control
    expected_loss_ship_rel: float  # as a fraction of the control posterior mean
    cred_interval_abs: tuple[float, float]
    cred_interval_rel: tuple[float, float]
    cred_mass: float

    def as_rows(self) -> list[list[str]]:
        return [
            ["metric", self.metric],
            ["model", f"{self.kind} / {self.prior}"],
            ["posterior mean (control)", f"{self.post_mean_control:.6g}"],
            ["posterior mean (treatment)", f"{self.post_mean_treatment:.6g}"],
            ["P(treatment > control)", f"{self.prob_treatment_better * 100:.2f}%"],
            [f"{self.cred_mass:.0%} credible interval (abs lift)",
             f"[{self.cred_interval_abs[0]:+.5g}, {self.cred_interval_abs[1]:+.5g}]"],
            [f"{self.cred_mass:.0%} credible interval (rel lift)",
             f"[{self.cred_interval_rel[0] * 100:+.2f}%, "
             f"{self.cred_interval_rel[1] * 100:+.2f}%]"],
            ["expected loss if we SHIP treatment",
             f"{self.expected_loss_ship:.3g} ({self.expected_loss_ship_rel * 100:.4f}% "
             "of control)"],
            ["expected loss if we KEEP control", f"{self.expected_loss_keep:.3g}"],
        ]


def _grid_weights(dist, n_points: int, tail: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid over the effective support of `dist` with normalised weights.

    A uniform grid (rather than equal-probability quantiles) keeps the outer
    product below a *product* measure, so joint functionals such as the ratio
    t/c can be evaluated by simple weighted quantiles.
    """
    lo, hi = dist.ppf(tail), dist.isf(tail)
    xs = np.linspace(lo, hi, n_points)
    w = dist.pdf(xs)
    w = w / w.sum()
    return xs, w


def _joint_quantiles(dist_c, dist_t, probs: np.ndarray,
                     n_points: int = 801) -> tuple[np.ndarray, np.ndarray]:
    """Weighted quantiles of the absolute and relative lift under independent posteriors."""
    xc, wc = _grid_weights(dist_c, n_points)
    xt, wt = _grid_weights(dist_t, n_points)
    weights = np.outer(wt, wc).ravel()
    diff = (xt[:, None] - xc[None, :]).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (xt[:, None] / xc[None, :] - 1.0).ravel()
    out = []
    for values in (diff, rel):
        order = np.argsort(values, kind="stable")
        v_sorted = values[order]
        cw = np.cumsum(weights[order])
        idx = np.searchsorted(cw, probs)
        idx = np.clip(idx, 0, v_sorted.size - 1)
        out.append(v_sorted[idx])
    return out[0], out[1]


def beta_binomial(x_c: int, n_c: int, x_t: int, n_t: int, *, metric: str = "conversion",
                  prior_a: float = 1.0, prior_b: float = 1.0,
                  cred_mass: float = 0.95) -> BayesResult:
    """Beta-Binomial posterior analysis of a rate metric.

    Prior Beta(1,1) by default = uniform, i.e. maximally non-committal; with tens of
    thousands of users the prior is irrelevant, but it keeps the posterior proper
    when an arm has zero conversions (a real day-1 situation).
    """
    a_c, b_c = prior_a + x_c, prior_b + (n_c - x_c)
    a_t, b_t = prior_a + x_t, prior_b + (n_t - x_t)
    post_c = stats.beta(a_c, b_c)
    post_t = stats.beta(a_t, b_t)
    m_c, m_t = a_c / (a_c + b_c), a_t / (a_t + b_t)

    lo = max(0.0, min(post_c.ppf(1e-12), post_t.ppf(1e-12)))
    hi = min(1.0, max(post_c.isf(1e-12), post_t.isf(1e-12)))

    # P(p_t > p_c) = int f_t(y) F_c(y) dy
    prob_better, _ = integrate.quad(
        lambda y: post_t.pdf(y) * post_c.cdf(y), lo, hi, limit=400)
    prob_better = float(np.clip(prob_better, 0.0, 1.0))

    # int_y^1 (x - y) f(x) dx = mean * SF_{a+1,b}(y) - y * SF_{a,b}(y)
    def _tail_excess(y: np.ndarray | float, a: float, b: float, mean: float):
        return mean * stats.beta.sf(y, a + 1, b) - y * stats.beta.sf(y, a, b)

    loss_ship, _ = integrate.quad(
        lambda y: post_t.pdf(y) * _tail_excess(y, a_c, b_c, m_c), lo, hi, limit=400)
    loss_keep, _ = integrate.quad(
        lambda y: post_c.pdf(y) * _tail_excess(y, a_t, b_t, m_t), lo, hi, limit=400)

    q = np.array([(1 - cred_mass) / 2, 1 - (1 - cred_mass) / 2])
    d_q, r_q = _joint_quantiles(post_c, post_t, q)
    return BayesResult(
        metric=metric, kind="rate", prior=f"Beta({prior_a:g},{prior_b:g})",
        post_mean_control=m_c, post_mean_treatment=m_t,
        prob_treatment_better=prob_better,
        expected_loss_ship=max(float(loss_ship), 0.0),
        expected_loss_keep=max(float(loss_keep), 0.0),
        expected_loss_ship_rel=max(float(loss_ship), 0.0) / m_c if m_c > 0 else float("nan"),
        cred_interval_abs=(float(d_q[0]), float(d_q[1])),
        cred_interval_rel=(float(r_q[0]), float(r_q[1])),
        cred_mass=cred_mass,
    )


def normal_normal(y_c: np.ndarray, y_t: np.ndarray, *, metric: str = "revenue_per_user",
                  cred_mass: float = 0.95) -> BayesResult:
    """Normal-Normal posterior for a continuous metric under a flat prior on the mean.

    With a flat prior and the CLT, the posterior for each arm's mean is
    N(sample mean, s^2/n).  That is the honest large-sample statement: we are NOT
    claiming the raw metric is Gaussian (revenue never is), only its mean.
    """
    y_c = np.asarray(y_c, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    if y_c.size < 2 or y_t.size < 2:
        raise ValueError("need at least 2 observations per arm")
    m_c, m_t = float(y_c.mean()), float(y_t.mean())
    se_c = math.sqrt(float(y_c.var(ddof=1)) / y_c.size)
    se_t = math.sqrt(float(y_t.var(ddof=1)) / y_t.size)
    sd = math.sqrt(se_c**2 + se_t**2)
    d = m_t - m_c

    prob_better = float(stats.norm.cdf(d / sd)) if sd > 0 else float(d > 0)
    # For X ~ N(m, s): E[max(X,0)] = m*Phi(m/s) + s*phi(m/s).
    loss_ship = -d * stats.norm.cdf(-d / sd) + sd * stats.norm.pdf(d / sd)
    loss_keep = d * stats.norm.cdf(d / sd) + sd * stats.norm.pdf(d / sd)

    q = np.array([(1 - cred_mass) / 2, 1 - (1 - cred_mass) / 2])
    d_q, r_q = _joint_quantiles(stats.norm(m_c, se_c), stats.norm(m_t, se_t), q)
    return BayesResult(
        metric=metric, kind="continuous", prior="flat on the mean (CLT posterior)",
        post_mean_control=m_c, post_mean_treatment=m_t,
        prob_treatment_better=prob_better,
        expected_loss_ship=max(float(loss_ship), 0.0),
        expected_loss_keep=max(float(loss_keep), 0.0),
        expected_loss_ship_rel=(max(float(loss_ship), 0.0) / m_c
                                if m_c > 0 else float("nan")),
        cred_interval_abs=(float(d_q[0]), float(d_q[1])),
        cred_interval_rel=(float(r_q[0]), float(r_q[1])),
        cred_mass=cred_mass,
    )


def format_bayes(res: BayesResult) -> str:
    from tabulate import tabulate

    return tabulate(res.as_rows(), headers=["posterior quantity", "value"],
                    tablefmt="github")
