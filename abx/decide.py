"""The decision layer: turn every statistic into one plain-English recommendation.

The rule order encodes what a competent experimentation reviewer does, in order:

  1. Trust the data at all?      -> SRM guard.  Fails => BLOCK, nothing else is read.
  2. Is the evidence real?       -> the ALWAYS-VALID p-value, not the naive one,
                                    because the team has been watching a dashboard.
  3. Is it big enough to matter? -> lower bound of the interval vs the practical MDE,
                                    plus the Bayesian expected loss from shipping.
  4. Is it safe to stop?         -> if not significant, has the experiment ruled out
                                    an effect worth having, or is it merely young?

Statistical significance alone never produces SHIP here.  A +0.2% lift whose interval
excludes zero is a real effect and still not worth the maintenance cost of a new code
path, so the practical MDE is a first-class input, and the expected-loss threshold is
the Bayesian cross-check that the downside tail is acceptable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bayes import BayesResult
from .frequentist import TestResult
from .guards import SRMResult
from .hte import SegmentScan
from .sequential import SequentialMonitor

SHIP = "SHIP"
DO_NOT_SHIP = "DO NOT SHIP"
KEEP_RUNNING = "KEEP RUNNING"
BLOCKED = "BLOCKED - SRM"


@dataclass
class DecisionConfig:
    """Business thresholds. These are policy, not statistics - so they are explicit."""

    alpha: float = 0.05
    practical_mde_rel: float = 0.02       # smallest relative lift worth shipping
    expected_loss_threshold_rel: float = 0.002  # 0.2% of control metric
    prob_better_threshold: float = 0.95
    srm_threshold: float = 0.001
    fdr_q: float = 0.10


@dataclass
class Recommendation:
    """Final verdict plus the audit trail that produced it."""

    verdict: str
    headline: str
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    narrative: str = ""
    source: str = "rules"

    def render(self) -> str:
        lines = [f"RECOMMENDATION: {self.verdict}", f"  {self.headline}", "", "Why:"]
        lines += [f"  - {r}" for r in self.reasons]
        if self.caveats:
            lines += ["", "Caveats:"] + [f"  - {c}" for c in self.caveats]
        if self.narrative:
            lines += ["", "Plain English:", *_wrap(self.narrative, 92)]
        return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return ["  " + ln for ln in textwrap.wrap(text, width=width)]


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def decide(*, srm: SRMResult, primary: TestResult, bayes: BayesResult,
           monitor: SequentialMonitor | None, config: DecisionConfig,
           segments: SegmentScan | None = None,
           secondary: list[TestResult] | None = None,
           planned_n_per_arm: int | None = None,
           llm=None) -> Recommendation:
    """Produce the verdict for one experiment."""
    reasons: list[str] = []
    caveats: list[str] = []
    secondary = secondary or []

    if not srm.passed:
        rec = Recommendation(
            verdict=BLOCKED,
            headline=("Sample-ratio mismatch: the realised split is "
                      f"{srm.observed_share[srm.arms.index(srm.worst_arm)]:.4f} for "
                      f"'{srm.worst_arm}' against a designed "
                      f"{srm.expected_share[srm.arms.index(srm.worst_arm)]:.4f}."),
            reasons=[
                srm.verdict(),
                "The arms are no longer exchangeable, so every lift estimate below is "
                "confounded by whatever broke the assignment.",
                "Fix the bucketing/logging defect and rerun; do not read the metrics.",
            ],
            caveats=["No conclusion about the feature can be drawn from this run."],
        )
        rec.narrative = _narrate(rec, primary, bayes, monitor, config, llm)
        return rec

    reasons.append(srm.verdict())

    av_p = float(monitor.always_valid_p[-1]) if monitor is not None else primary.p_value
    p_label = "always-valid p" if monitor is not None else "p"
    evidence = av_p <= config.alpha
    direction_positive = primary.abs_lift > 0
    ci_low, ci_high = primary.rel_ci
    loss_ok = bayes.expected_loss_ship_rel <= config.expected_loss_threshold_rel
    prob_ok = bayes.prob_treatment_better >= config.prob_better_threshold
    practical = ci_low >= config.practical_mde_rel

    reasons.append(
        f"{primary.metric}: {_pct(primary.rel_lift)} relative "
        f"({primary.abs_lift:+.5g} absolute), 95% CI [{_pct(ci_low)}, {_pct(ci_high)}], "
        f"fixed-horizon p={primary.p_value:.3g}, {p_label}={av_p:.3g}.")
    reasons.append(
        f"Posterior: P(treatment > control) = {bayes.prob_treatment_better * 100:.2f}%, "
        f"expected loss from shipping = {bayes.expected_loss_ship:.3g} "
        f"({bayes.expected_loss_ship_rel * 100:.4f}% of control, threshold "
        f"{config.expected_loss_threshold_rel * 100:.2f}%).")

    if monitor is not None and monitor.naive_first_cross is not None:
        cross = monitor.naive_first_cross
        if monitor.msprt_first_cross is None and monitor.obf_first_cross is None:
            caveats.append(
                f"A naive dashboard would have called this at look {cross + 1} "
                f"(naive p={monitor.naive_p[cross]:.4f}); neither the alpha-spending "
                "boundary nor the always-valid p-value ever crossed. That gap is the "
                "peeking false positive this platform exists to prevent.")

    if evidence and not direction_positive:
        verdict = DO_NOT_SHIP
        headline = (f"The treatment HURTS {primary.metric} by {_pct(abs(primary.rel_lift))} "
                    "and the evidence survives continuous monitoring.")
        reasons.append("Effect is negative with always-valid significance: this is a "
                       "confirmed regression, not noise.")
    elif evidence and direction_positive and practical and (loss_ok or prob_ok):
        verdict = SHIP
        headline = (f"{primary.metric} is up {_pct(primary.rel_lift)} and the whole "
                    f"interval clears the {_pct(config.practical_mde_rel)} bar worth shipping.")
        reasons.append(
            f"Lower confidence bound {_pct(ci_low)} exceeds the practical MDE "
            f"{_pct(config.practical_mde_rel)}, so even the pessimistic reading is a win.")
        reasons.append("Expected loss from shipping is below the risk threshold, so the "
                       "downside tail is affordable.")
    elif evidence and direction_positive and not practical:
        verdict = DO_NOT_SHIP
        headline = (f"{primary.metric} is statistically up {_pct(primary.rel_lift)} but the "
                    "interval does not clear the bar for a change worth maintaining.")
        reasons.append(
            f"Lower bound {_pct(ci_low)} is under the practical MDE "
            f"{_pct(config.practical_mde_rel)}: real, but too small to justify the change.")
    elif not evidence and _ruled_out(ci_high, config.practical_mde_rel) and \
            _sample_complete(primary, planned_n_per_arm):
        verdict = DO_NOT_SHIP
        headline = (f"No effect, and the experiment is now precise enough to rule out a "
                    f"{_pct(config.practical_mde_rel)} lift.")
        reasons.append(
            f"Upper bound {_pct(ci_high)} sits below the practical MDE "
            f"{_pct(config.practical_mde_rel)}: an effect worth having has been excluded.")
    elif not evidence and _sample_complete(primary, planned_n_per_arm):
        verdict = DO_NOT_SHIP
        headline = ("Inconclusive at the planned sample size: no significant effect and "
                    "the interval still admits effects worth having.")
        reasons.append(
            f"Planned sample reached with {p_label}={av_p:.3g}; the interval "
            f"[{_pct(ci_low)}, {_pct(ci_high)}] is too wide to be decisive.")
        caveats.append("Extending the run is only worthwhile if the extra traffic is "
                       "cheaper than the next experiment in the queue.")
    else:
        verdict = KEEP_RUNNING
        need = _extra_units(primary, planned_n_per_arm)
        headline = ("Too early to call: no always-valid significance yet and the planned "
                    "sample is not complete.")
        reasons.append(
            f"{p_label}={av_p:.3g} has not crossed alpha={config.alpha:g}; because the "
            "monitoring is always-valid, continuing costs no false-positive budget.")
        if need:
            reasons.append(f"About {need:,} more units per arm to reach the plan.")

    for sec in secondary:
        flag = "significant" if sec.significant else "not significant"
        reasons.append(f"Secondary {sec.metric}: {_pct(sec.rel_lift)} "
                       f"(p={sec.p_value:.3g}, {flag}).")
        if sec.significant and sec.abs_lift < 0 < primary.abs_lift:
            caveats.append(f"Guardrail conflict: {sec.metric} moved the wrong way.")

    if segments is not None and segments.n_hypotheses:
        if segments.n_significant_bh:
            top = segments.discoveries.iloc[0]
            reasons.append(
                f"Segment scan: {segments.n_significant_bh} of {segments.n_hypotheses} "
                f"interactions survive BH at q={segments.q:g}; strongest is "
                f"{top['dimension']}={top['level']} "
                f"(lift {top['abs_lift']:+.4f} inside vs {top['lift_outside']:+.4f} "
                f"outside, BH p={top['bh_adjusted_p']:.3g}).")
            caveats.append("Segment findings are exploratory: confirm them in a "
                           "dedicated follow-up experiment before targeting the rollout.")
        else:
            reasons.append(
                f"Segment scan: {segments.n_significant_uncorrected} of "
                f"{segments.n_hypotheses} interactions look significant uncorrected, "
                f"0 survive Benjamini-Hochberg at q={segments.q:g} - i.e. the apparent "
                "heterogeneity is multiple-testing noise.")

    rec = Recommendation(verdict=verdict, headline=headline, reasons=reasons,
                         caveats=caveats)
    rec.narrative = _narrate(rec, primary, bayes, monitor, config, llm)
    return rec


def _ruled_out(ci_high: float, mde: float) -> bool:
    return ci_high < mde


def _sample_complete(primary: TestResult, planned_n_per_arm: int | None) -> bool:
    if planned_n_per_arm is None:
        return True
    return min(primary.n_control, primary.n_treatment) >= planned_n_per_arm


def _extra_units(primary: TestResult, planned_n_per_arm: int | None) -> int | None:
    if planned_n_per_arm is None:
        return None
    have = min(primary.n_control, primary.n_treatment)
    return max(planned_n_per_arm - have, 0)


# --------------------------------------------------------------------------- #
# Narrative: deterministic by default, LLM-polished when credentials exist.
# --------------------------------------------------------------------------- #

def _deterministic_narrative(rec: Recommendation, primary: TestResult,
                             bayes: BayesResult, monitor: SequentialMonitor | None,
                             config: DecisionConfig) -> str:
    n_total = primary.n_control + primary.n_treatment
    if rec.verdict == BLOCKED:
        return (
            f"Stop here. The traffic split is broken, so the {n_total:,} users in this "
            "run are not a fair comparison and no number below can be trusted. Nothing "
            "about the feature has been learned yet; fix the assignment defect, verify "
            "the split on a fresh A/A run, and repeat the experiment.")
    parts = [
        f"Across {n_total:,} users, {primary.metric} moved from "
        f"{primary.est_control:.4g} in control to {primary.est_treatment:.4g} in "
        f"treatment, a relative change of {_pct(primary.rel_lift)}."
    ]
    if monitor is not None:
        parts.append(
            f"After {monitor.looks.size} interim looks the always-valid p-value is "
            f"{monitor.always_valid_p[-1]:.3g}, so this reading is safe even though the "
            "experiment was watched every day.")
    parts.append(
        f"The posterior puts {bayes.prob_treatment_better * 100:.1f}% probability on the "
        f"treatment being better, and shipping it risks on average "
        f"{bayes.expected_loss_ship_rel * 100:.3f}% of the control metric if we are wrong.")
    if rec.verdict == SHIP:
        parts.append(
            f"Because the pessimistic end of the interval ({_pct(primary.rel_ci[0])}) is "
            f"still above the {_pct(config.practical_mde_rel)} we said was worth the "
            "engineering cost, the recommendation is to ship.")
    elif rec.verdict == KEEP_RUNNING:
        parts.append(
            "The evidence has not crossed the bar yet. Because the monitoring is "
            "always-valid, letting it run costs nothing statistically - so keep it "
            "running rather than calling it now.")
    else:
        parts.append(
            f"The result does not clear the bar for a change worth maintaining "
            f"({_pct(config.practical_mde_rel)} relative), so the recommendation is not "
            "to ship this variant.")
    if rec.caveats:
        parts.append("Worth flagging: " + rec.caveats[0])
    return " ".join(parts)


def _narrate(rec: Recommendation, primary: TestResult, bayes: BayesResult,
             monitor: SequentialMonitor | None, config: DecisionConfig, llm) -> str:
    """Deterministic narrative; an available LLM only rephrases the same facts."""
    base = _deterministic_narrative(rec, primary, bayes, monitor, config)
    if llm is None:
        rec.source = "rules"
        return base
    prompt = (
        "Rewrite the following experiment readout as one tight paragraph for a product "
        "manager. Keep every number exactly as given, do not add numbers, do not change "
        "the recommendation, and stay under 130 words.\n\n"
        f"RECOMMENDATION: {rec.verdict}\nHEADLINE: {rec.headline}\n"
        f"FACTS: {base}")
    try:
        out = llm.complete(prompt, system="You are a careful experimentation analyst.")
    except Exception:
        out = ""
    if not out:
        rec.source = "rules (LLM unavailable)"
        return base
    rec.source = "llm-polished"
    return out
