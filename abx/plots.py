"""Figures for the analysis report.  Agg backend: these run headless on CI/Windows."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .cuped import CUPEDResult  # noqa: E402
from .power import power_curve  # noqa: E402
from .sequential import PeekingSimulation, SequentialMonitor  # noqa: E402


def plot_sequential(monitor: SequentialMonitor, out_path: Path, title: str) -> Path:
    """Interim z-trajectory against the naive line and the alpha-spending boundary."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = monitor.info_fractions
    ax.plot(t, monitor.boundaries.z_boundaries, "o-", color="#c0392b",
            label="O'Brien-Fleming boundary")
    ax.plot(t, -monitor.boundaries.z_boundaries, "o-", color="#c0392b")
    z_crit = stats.norm.ppf(1 - monitor.alpha / 2)
    ax.axhline(z_crit, color="#7f8c8d", ls="--", label=f"naive +/-{z_crit:.2f}")
    ax.axhline(-z_crit, color="#7f8c8d", ls="--")
    ax.plot(t, monitor.z, "s-", color="#2c3e50", lw=2, label="observed z")
    ax.set_xlabel("information fraction")
    ax.set_ylabel("z statistic")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_peeking(sim: PeekingSimulation, out_path: Path) -> Path:
    """Bar chart of the false-positive rate of each monitoring policy."""
    labels = ["fixed\nhorizon", f"naive\n{sim.n_looks} peeks", "OBF\nspending",
              "mixture\nSPRT"]
    values = [sim.fixed_horizon_rate, sim.naive_rate, sim.obf_rate, sim.msprt_rate]
    colors = ["#7f8c8d", "#c0392b", "#27ae60", "#2980b9"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(sim.alpha, color="black", ls="--", lw=1,
               label=f"nominal alpha = {sim.alpha:g}")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylabel("false-positive rate")
    ax.set_ylim(0, max(values) * 1.25 + 0.02)
    ax.set_title(f"Repeated peeking under the null ({sim.n_sims:,} simulated A/A tests)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_posteriors(a_c: float, b_c: float, a_t: float, b_t: float,
                    out_path: Path, title: str) -> Path:
    """Beta posteriors for both arms plus the posterior of the absolute lift."""
    post_c, post_t = stats.beta(a_c, b_c), stats.beta(a_t, b_t)
    lo = min(post_c.ppf(1e-6), post_t.ppf(1e-6))
    hi = max(post_c.isf(1e-6), post_t.isf(1e-6))
    xs = np.linspace(lo, hi, 600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(xs, post_c.pdf(xs), color="#7f8c8d", label="control")
    ax1.fill_between(xs, post_c.pdf(xs), color="#7f8c8d", alpha=0.25)
    ax1.plot(xs, post_t.pdf(xs), color="#2980b9", label="treatment")
    ax1.fill_between(xs, post_t.pdf(xs), color="#2980b9", alpha=0.25)
    ax1.set_title("Beta-Binomial posteriors")
    ax1.set_xlabel("conversion rate")
    ax1.legend(fontsize=8)

    # Posterior of the difference by numerical convolution on a common grid.
    n = 1200
    xc = np.linspace(post_c.ppf(1e-9), post_c.isf(1e-9), n)
    xt = np.linspace(post_t.ppf(1e-9), post_t.isf(1e-9), n)
    wc = post_c.pdf(xc); wc /= wc.sum()
    wt = post_t.pdf(xt); wt /= wt.sum()
    diff = (xt[:, None] - xc[None, :]).ravel()
    w = np.outer(wt, wc).ravel()
    edges = np.linspace(diff.min(), diff.max(), 160)
    hist, _ = np.histogram(diff, bins=edges, weights=w)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax2.bar(centres, hist, width=edges[1] - edges[0], color="#27ae60", alpha=0.75)
    ax2.axvline(0.0, color="black", ls="--", lw=1)
    ax2.set_title("Posterior of the absolute lift")
    ax2.set_xlabel("treatment - control")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_power(base_rate: float, n_per_arm: int, cuped: CUPEDResult,
               out_path: Path) -> Path:
    """Power curves with and without the CUPED variance reduction."""
    lifts, p_raw = power_curve(base_rate, n_per_arm)
    _, p_cuped = power_curve(base_rate, n_per_arm,
                             variance_reduction=cuped.variance_reduction)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(lifts * 100, p_raw, color="#7f8c8d", lw=2, label="raw metric")
    ax.plot(lifts * 100, p_cuped, color="#27ae60", lw=2,
            label=f"with CUPED ({cuped.variance_reduction * 100:.1f}% var reduction)")
    ax.axhline(0.8, color="black", ls="--", lw=1, label="80% power")
    ax.set_xlabel("true relative lift (%)")
    ax.set_ylabel("power")
    ax.set_title(f"Power at n={n_per_arm:,} per arm, base rate {base_rate:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
