"""Sample-ratio-mismatch (SRM) guard - the first thing to check, before any p-value.

WHY this blocks everything downstream: if the realised traffic split differs from the
designed split by more than chance, the randomisation is broken (bot filtering,
redirect latency, a bugged bucketing hash, logging loss on one arm...).  When that
happens the two arms are no longer exchangeable, so EVERY estimate below - lift,
posterior, CUPED, segment scan - is confounded by whatever caused the imbalance.
Reporting a "winner" from an SRM-broken experiment is worse than reporting nothing.

The conventional threshold is p < 0.001 rather than 0.05: an SRM alarm is expensive
to investigate and the chi-square statistic is enormous when the split really is
broken, so a strict threshold keeps the false-alarm rate negligible without losing
sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    """Chi-square goodness-of-fit test of the observed vs designed traffic split."""

    arms: list[str]
    observed: np.ndarray
    expected: np.ndarray
    observed_share: np.ndarray
    expected_share: np.ndarray
    chi2: float
    dof: int
    p_value: float
    threshold: float

    @property
    def passed(self) -> bool:
        return self.p_value >= self.threshold

    @property
    def worst_arm(self) -> str:
        """Arm furthest from its designed share; ties go to the OVER-represented arm.

        With two arms the deviations are equal and opposite, so the tie-break matters:
        naming the arm that received too much traffic is what points an engineer at the
        bucketing or redirect bug.
        """
        resid = (self.observed - self.expected) / np.sqrt(self.expected)
        largest = np.abs(resid)
        candidates = np.flatnonzero(largest >= largest.max() - 1e-9)
        return self.arms[int(candidates[np.argmax(resid[candidates])])]

    def table(self) -> str:
        from tabulate import tabulate

        rows = []
        for i, arm in enumerate(self.arms):
            delta = self.observed[i] - self.expected[i]
            rows.append([arm, int(self.observed[i]), f"{self.expected[i]:.1f}",
                         f"{delta:+.1f}", f"{self.observed_share[i]:.4f}",
                         f"{self.expected_share[i]:.4f}"])
        return tabulate(rows, headers=["arm", "observed", "expected", "delta",
                                       "observed share", "designed share"],
                        tablefmt="github")

    def verdict(self) -> str:
        if self.passed:
            return (f"SRM check PASSED (chi2={self.chi2:.2f}, dof={self.dof}, "
                    f"p={self.p_value:.4g} >= {self.threshold:g})")
        return (f"SRM check FAILED (chi2={self.chi2:.2f}, dof={self.dof}, "
                f"p={self.p_value:.4g} < {self.threshold:g}); the '{self.worst_arm}' arm "
                "is off its designed share - randomisation is suspect")


def srm_check(counts: dict[str, int], designed_split: dict[str, float],
              threshold: float = 0.001) -> SRMResult:
    """Chi-square SRM test.  `designed_split` need not be normalised."""
    arms = list(counts.keys())
    missing = [a for a in arms if a not in designed_split]
    if missing:
        raise ValueError(f"no designed split for arm(s): {missing}")
    obs = np.array([counts[a] for a in arms], dtype=float)
    share = np.array([designed_split[a] for a in arms], dtype=float)
    if np.any(share <= 0):
        raise ValueError("designed split shares must be positive")
    share = share / share.sum()
    total = obs.sum()
    if total <= 0:
        raise ValueError("no units assigned")
    exp = total * share
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    dof = len(arms) - 1
    p = float(stats.chi2.sf(chi2, dof))
    return SRMResult(arms=arms, observed=obs, expected=exp,
                     observed_share=obs / total, expected_share=share,
                     chi2=chi2, dof=dof, p_value=p, threshold=threshold)
