"""Heterogeneous treatment effect scan with Benjamini-Hochberg FDR control.

Segment fishing is the most common way an experiment platform manufactures false
discoveries: slice 12 ways, find the one slice with p < 0.05, ship it as "the feature
works great on mobile".  Two corrections are applied here.

1. The right hypothesis is the INTERACTION, not the within-segment effect.  A segment
   can show a significant lift simply because the overall effect is real; the
   interesting claim is that the effect DIFFERS inside the segment.  So for each
   level we compare the lift inside the level with the lift outside it:
       z = (d_in - d_out) / sqrt(se_in^2 + se_out^2)
   The two groups are disjoint, so the variances add.

2. Benjamini-Hochberg controls the false discovery rate across all levels scanned.
   BH rather than Bonferroni because segment discoveries are exploratory hypotheses
   for a follow-up experiment - controlling the expected *proportion* of false leads
   at 10% is the useful guarantee, and it is far more powerful than FWER control.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .frequentist import two_proportion_ztest


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.10
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (rejected mask, BH-adjusted p-values) at FDR level `q`."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    below = ranked <= q * ranks / m
    k = int(np.flatnonzero(below).max()) + 1 if below.any() else 0
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k]] = True
    # Step-up adjusted p-values (monotone from the largest rank down).
    adj_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(adj_sorted, 0.0, 1.0)
    return rejected, adjusted


@dataclass
class SegmentScan:
    """Result of scanning every level of every segment dimension."""

    frame: pd.DataFrame
    q: float
    n_hypotheses: int
    n_significant_uncorrected: int
    n_significant_bh: int
    metric: str

    @property
    def discoveries(self) -> pd.DataFrame:
        return self.frame[self.frame["bh_significant"]].copy()

    def table(self, top: int = 12) -> str:
        from tabulate import tabulate

        cols = ["dimension", "level", "n", "rate_control", "rate_treatment",
                "abs_lift", "rel_lift", "lift_outside", "interaction_z",
                "interaction_p", "bh_adjusted_p", "bh_significant"]
        view = self.frame.sort_values("interaction_p").head(top)[cols].copy()
        rows = []
        for _, r in view.iterrows():
            rows.append([
                r["dimension"], r["level"], f"{int(r['n']):,}",
                f"{r['rate_control']:.4f}", f"{r['rate_treatment']:.4f}",
                f"{r['abs_lift']:+.4f}", f"{r['rel_lift'] * 100:+.2f}%",
                f"{r['lift_outside']:+.4f}", f"{r['interaction_z']:+.2f}",
                f"{r['interaction_p']:.4g}", f"{r['bh_adjusted_p']:.4g}",
                "YES" if r["bh_significant"] else "no",
            ])
        return tabulate(rows, headers=["dim", "level", "n", "rate C", "rate T",
                                       "abs lift", "rel lift", "lift outside",
                                       "inter z", "inter p", "BH p", "discovery"],
                        tablefmt="github")


def scan_segments(df: pd.DataFrame, dimensions: list[str], *, variant_col: str = "variant",
                  outcome_col: str = "converted", control: str = "control",
                  treatment: str = "treatment", q: float = 0.10,
                  min_n: int = 500) -> SegmentScan:
    """Scan every level of `dimensions` for a treatment-effect interaction.

    Levels with fewer than `min_n` units are skipped: the normal approximation for the
    interaction z is unreliable there and such levels only add noise to the FDR
    denominator.
    """
    is_t = (df[variant_col] == treatment).to_numpy()
    is_c = (df[variant_col] == control).to_numpy()
    y = df[outcome_col].to_numpy(dtype=float)

    records: list[dict] = []
    for dim in dimensions:
        levels = df[dim].to_numpy()
        for level in pd.unique(df[dim]):
            inside = levels == level
            n_in = int(inside.sum())
            if n_in < min_n:
                continue
            stats_in = _arm_stats(y, is_c, is_t, inside)
            stats_out = _arm_stats(y, is_c, is_t, ~inside)
            if stats_in is None or stats_out is None:
                continue
            d_in, se_in, p_c_in, p_t_in, n_c_in, n_t_in = stats_in
            d_out, se_out, *_ = stats_out
            se_int = float(np.sqrt(se_in**2 + se_out**2))
            z_int = (d_in - d_out) / se_int if se_int > 0 else 0.0
            p_int = float(2 * stats.norm.sf(abs(z_int)))
            within = two_proportion_ztest(int(round(p_c_in * n_c_in)), n_c_in,
                                          int(round(p_t_in * n_t_in)), n_t_in,
                                          metric=f"{dim}={level}")
            records.append({
                "dimension": dim, "level": str(level), "n": n_in,
                "n_control": n_c_in, "n_treatment": n_t_in,
                "rate_control": p_c_in, "rate_treatment": p_t_in,
                "abs_lift": d_in, "rel_lift": within.rel_lift,
                "within_p": within.p_value,
                "lift_outside": d_out, "interaction_z": z_int, "interaction_p": p_int,
            })

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["dimension", "level", "n", "n_control",
                                      "n_treatment", "rate_control", "rate_treatment",
                                      "abs_lift", "rel_lift", "within_p",
                                      "lift_outside", "interaction_z", "interaction_p",
                                      "bh_adjusted_p", "bh_significant"])
        return SegmentScan(frame=frame, q=q, n_hypotheses=0,
                           n_significant_uncorrected=0, n_significant_bh=0,
                           metric=outcome_col)
    rejected, adjusted = benjamini_hochberg(frame["interaction_p"].to_numpy(), q=q)
    frame["bh_adjusted_p"] = adjusted
    frame["bh_significant"] = rejected
    return SegmentScan(
        frame=frame.sort_values("interaction_p").reset_index(drop=True), q=q,
        n_hypotheses=len(frame),
        n_significant_uncorrected=int((frame["interaction_p"] < 0.05).sum()),
        n_significant_bh=int(rejected.sum()), metric=outcome_col,
    )


def _arm_stats(y: np.ndarray, is_c: np.ndarray, is_t: np.ndarray, mask: np.ndarray):
    """Per-arm rates, lift and its unpooled SE inside `mask`; None if too thin."""
    c = mask & is_c
    t = mask & is_t
    n_c, n_t = int(c.sum()), int(t.sum())
    if n_c < 30 or n_t < 30:
        return None
    p_c = float(y[c].mean())
    p_t = float(y[t].mean())
    se = float(np.sqrt(max(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t, 1e-300)))
    return p_t - p_c, se, p_c, p_t, n_c, n_t
