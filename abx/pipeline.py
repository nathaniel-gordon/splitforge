"""End-to-end analysis of one experiment, in the order a reviewer would run it.

The order is load-bearing: the SRM guard runs FIRST and short-circuits the verdict,
because reporting lifts from an experiment with broken randomisation is the failure
mode that costs companies real money.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .bayes import BayesResult, beta_binomial, format_bayes, normal_normal
from .cuped import CUPEDResult, apply_cuped
from .datagen import SEGMENT_DIMENSIONS, daily_cumulative
from .decide import DecisionConfig, Recommendation, decide
from .frequentist import TestResult, format_results, two_proportion_ztest, welch_ttest
from .guards import SRMResult, srm_check
from .hte import SegmentScan, scan_segments
from .power import PowerPlan, achieved_power_proportion, mde_proportion, \
    sample_size_proportion
from .sequential import SequentialMonitor, monitor_experiment


@dataclass
class AnalysisResult:
    """Everything the platform computed for one experiment."""

    name: str
    n_users: int
    srm: SRMResult
    conversion: TestResult
    revenue: TestResult
    bayes_conversion: BayesResult
    bayes_revenue: BayesResult
    cuped: CUPEDResult
    monitor: SequentialMonitor
    segments: SegmentScan
    plan: PowerPlan
    mde_rel: float
    achieved_power: float
    achieved_power_cuped: float
    recommendation: Recommendation
    daily: pd.DataFrame
    posterior_params: tuple[float, float, float, float]
    config: DecisionConfig = field(default_factory=DecisionConfig)

    def report(self, include_segments: bool = True) -> str:
        """Full human-readable readout."""
        sep = "=" * 96
        blocks = [
            sep,
            f"EXPERIMENT: {self.name}   |   {self.n_users:,} users   |   "
            f"alpha={self.config.alpha:g}, practical MDE="
            f"{self.config.practical_mde_rel * 100:.1f}%",
            sep,
            "",
            "[1] SAMPLE RATIO MISMATCH GUARD",
            self.srm.table(),
            self.srm.verdict(),
            "",
            "[2] FREQUENTIST TESTS",
            format_results([self.conversion, self.revenue,
                            self.cuped.adjusted_test]),
            "",
            "[3] BAYESIAN POSTERIOR - conversion",
            format_bayes(self.bayes_conversion),
            "",
            "[3b] BAYESIAN POSTERIOR - revenue per user",
            format_bayes(self.bayes_revenue),
            "",
            "[4] CUPED VARIANCE REDUCTION",
            self.cuped.table(),
            "",
            "[5] SEQUENTIAL MONITORING (daily interim looks)",
            self.monitor.table(),
            self._sequential_note(),
            "",
            "[6] POWER / PLANNING",
            self.plan.table(),
            f"MDE at the realised sample size ({self.conversion.n_control:,} per arm): "
            f"{self.mde_rel * 100:.2f}% relative "
            f"({self.mde_rel * self.conversion.est_control:.5f} absolute)",
            f"Power against the observed lift: {self.achieved_power * 100:.1f}% raw, "
            f"{self.achieved_power_cuped * 100:.1f}% with the measured CUPED reduction",
        ]
        if include_segments:
            blocks += [
                "",
                "[7] HETEROGENEOUS TREATMENT EFFECTS "
                f"(BH FDR q={self.segments.q:g}, {self.segments.n_hypotheses} hypotheses)",
                self.segments.table(),
                f"uncorrected significant: {self.segments.n_significant_uncorrected} | "
                f"survive Benjamini-Hochberg: {self.segments.n_significant_bh}",
            ]
        blocks += ["", "[8] DECISION", self.recommendation.render(), sep]
        return "\n".join(blocks)

    def _sequential_note(self) -> str:
        m = self.monitor
        bits = [f"mixture-SPRT mixing scale tau={m.tau:.5f} (absolute rate units)"]
        naive = m.naive_first_cross
        obf = m.obf_first_cross
        msprt = m.msprt_first_cross
        bits.append(f"first crossing -> naive: {'look ' + str(naive + 1) if naive is not None else 'never'}"
                    f" | OBF: {'look ' + str(obf + 1) if obf is not None else 'never'}"
                    f" | mSPRT: {'look ' + str(msprt + 1) if msprt is not None else 'never'}")
        if naive is not None and obf is None and msprt is None:
            bits.append("=> the naive dashboard would have declared a winner that neither "
                        "valid procedure supports.")
        return "\n".join(bits)


def analyze(df: pd.DataFrame, *, name: str = "experiment",
            designed_split: dict[str, float] | None = None,
            config: DecisionConfig | None = None,
            dimensions: list[str] | None = None,
            planned_n_per_arm: int | None = None,
            plan_mde_rel: float = 0.03,
            llm=None) -> AnalysisResult:
    """Run the whole battery on a user-level experiment log."""
    config = config or DecisionConfig()
    designed_split = designed_split or {"control": 0.5, "treatment": 0.5}
    dimensions = dimensions or SEGMENT_DIMENSIONS

    counts = df["variant"].value_counts().to_dict()
    srm = srm_check({k: int(counts.get(k, 0)) for k in designed_split},
                    designed_split, threshold=config.srm_threshold)

    ctrl = df[df["variant"] == "control"]
    treat = df[df["variant"] == "treatment"]
    x_c, n_c = int(ctrl["converted"].sum()), len(ctrl)
    x_t, n_t = int(treat["converted"].sum()), len(treat)

    conversion = two_proportion_ztest(x_c, n_c, x_t, n_t, alpha=config.alpha,
                                      metric="conversion")
    revenue = welch_ttest(ctrl["revenue"].to_numpy(), treat["revenue"].to_numpy(),
                          alpha=config.alpha, metric="revenue_per_user")
    bayes_conv = beta_binomial(x_c, n_c, x_t, n_t, metric="conversion")
    bayes_rev = normal_normal(ctrl["revenue"].to_numpy(), treat["revenue"].to_numpy(),
                              metric="revenue_per_user")
    cuped = apply_cuped(ctrl["revenue"].to_numpy(), ctrl["pre_revenue"].to_numpy(),
                        treat["revenue"].to_numpy(), treat["pre_revenue"].to_numpy(),
                        alpha=config.alpha)

    daily = daily_cumulative(df)
    monitor = monitor_experiment(daily["x_control"].to_numpy(),
                                 daily["n_control"].to_numpy(),
                                 daily["x_treatment"].to_numpy(),
                                 daily["n_treatment"].to_numpy(),
                                 alpha=config.alpha)

    segments = scan_segments(df, dimensions, q=config.fdr_q)

    base_rate = conversion.est_control
    plan = sample_size_proportion(base_rate, plan_mde_rel, alpha=config.alpha,
                                  power=0.80, daily_traffic=len(df) / df["day"].max())
    mde = mde_proportion(base_rate, min(n_c, n_t), alpha=config.alpha, power=0.80)
    power_raw = achieved_power_proportion(base_rate, conversion.rel_lift, min(n_c, n_t),
                                          alpha=config.alpha)
    power_cuped = achieved_power_proportion(base_rate, conversion.rel_lift,
                                            min(n_c, n_t), alpha=config.alpha,
                                            variance_reduction=cuped.variance_reduction)

    rec = decide(srm=srm, primary=conversion, bayes=bayes_conv, monitor=monitor,
                 config=config, segments=segments,
                 secondary=[cuped.adjusted_test],
                 planned_n_per_arm=planned_n_per_arm, llm=llm)

    return AnalysisResult(
        name=name, n_users=len(df), srm=srm, conversion=conversion, revenue=revenue,
        bayes_conversion=bayes_conv, bayes_revenue=bayes_rev, cuped=cuped,
        monitor=monitor, segments=segments, plan=plan, mde_rel=mde,
        achieved_power=power_raw, achieved_power_cuped=power_cuped,
        recommendation=rec, daily=daily,
        posterior_params=(1.0 + x_c, 1.0 + n_c - x_c, 1.0 + x_t, 1.0 + n_t - x_t),
        config=config,
    )


def write_artifacts(result: AnalysisResult, out_dir: Path,
                    peeking=None) -> list[Path]:
    """Persist the report, the segment table and the figures."""
    from .plots import plot_peeking, plot_posteriors, plot_power, plot_sequential

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    report_path = out_dir / f"{result.name}_report.txt"
    report_path.write_text(result.report(), encoding="utf-8")
    paths.append(report_path)

    seg_path = out_dir / f"{result.name}_segments.csv"
    result.segments.frame.to_csv(seg_path, index=False)
    paths.append(seg_path)

    daily_path = out_dir / f"{result.name}_daily.csv"
    daily = result.daily.copy()
    daily["z"] = result.monitor.z
    daily["naive_p"] = result.monitor.naive_p
    daily["obf_z_boundary"] = result.monitor.boundaries.z_boundaries
    daily["always_valid_p"] = result.monitor.always_valid_p
    daily.to_csv(daily_path, index=False)
    paths.append(daily_path)

    paths.append(plot_sequential(result.monitor, out_dir / f"{result.name}_sequential.png",
                                 f"{result.name}: interim monitoring"))
    a_c, b_c, a_t, b_t = result.posterior_params
    paths.append(plot_posteriors(a_c, b_c, a_t, b_t,
                                 out_dir / f"{result.name}_posterior.png",
                                 f"{result.name}: conversion posteriors"))
    paths.append(plot_power(result.conversion.est_control,
                            result.conversion.n_control, result.cuped,
                            out_dir / f"{result.name}_power.png"))
    if peeking is not None:
        paths.append(plot_peeking(peeking, out_dir / "peeking_false_positive_rate.png"))
    return paths


def truncate_to_day(df: pd.DataFrame, day: int) -> pd.DataFrame:
    """Everything observed up to and including `day` - used to show an early look."""
    return df[df["day"] <= day].reset_index(drop=True)


def summarise(results: list[AnalysisResult]) -> str:
    """One-line-per-experiment scoreboard."""
    from tabulate import tabulate

    rows = []
    for r in results:
        rows.append([
            r.name, f"{r.n_users:,}",
            "PASS" if r.srm.passed else "FAIL",
            f"{r.conversion.rel_lift * 100:+.2f}%",
            f"{r.conversion.p_value:.3g}",
            f"{r.monitor.always_valid_p[-1]:.3g}",
            f"{r.bayes_conversion.prob_treatment_better * 100:.1f}%",
            f"{r.cuped.variance_reduction * 100:.1f}%",
            r.segments.n_significant_bh,
            r.recommendation.verdict,
        ])
    return tabulate(rows, headers=["experiment", "users", "SRM", "conv rel lift",
                                   "fixed p", "always-valid p", "P(T>C)",
                                   "CUPED var red", "BH segs", "verdict"],
                    tablefmt="github")


def arm_arrays(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Control/treatment arrays for one column - small helper used by the CLI."""
    return (df.loc[df["variant"] == "control", column].to_numpy(),
            df.loc[df["variant"] == "treatment", column].to_numpy())
