"""Smoke test with real statistical thresholds.

Run:  python tests/test_smoke.py

Every assertion below is a claim about correctness that would break if the maths
broke: closed-form identities (expected-loss balance, Welch agreement with scipy,
sample-size/MDE round trip), published reference values (Lan-DeMets O'Brien-Fleming
boundaries), and end-to-end ground truth from the seeded generator.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abx.bayes import beta_binomial, normal_normal  # noqa: E402
from abx.cuped import apply_cuped  # noqa: E402
from abx.datagen import SEGMENT_DIMENSIONS, generate  # noqa: E402
from abx.decide import DecisionConfig  # noqa: E402
from abx.frequentist import two_proportion_ztest, welch_ttest  # noqa: E402
from abx.guards import srm_check  # noqa: E402
from abx.hte import benjamini_hochberg  # noqa: E402
from abx.pipeline import analyze, truncate_to_day  # noqa: E402
from abx.power import (achieved_power_proportion, mde_proportion,  # noqa: E402
                       sample_size_proportion)
from abx.sequential import (alpha_spending_boundaries,  # noqa: E402
                            always_valid_pvalues, peeking_simulation)

SEED = 42


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{label} FAILED {detail}")


def test_frequentist_matches_scipy() -> None:
    """Welch t-test must reproduce scipy exactly; z-test must match the normal tail."""
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 3.0, 5000)
    b = rng.normal(10.4, 4.0, 4000)
    ours = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    check("welch t statistic", abs(ours.statistic - ref.statistic) < 1e-9,
          f"{ours.statistic} vs {ref.statistic}")
    check("welch p-value", abs(ours.p_value - ref.pvalue) < 1e-12,
          f"{ours.p_value} vs {ref.pvalue}")

    z = two_proportion_ztest(1000, 10000, 1100, 10000)
    p_pool = 2100 / 20000
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / 10000 + 1 / 10000))
    check("z statistic", abs(z.statistic - 0.01 / se) < 1e-12)
    check("z CI brackets the point estimate",
          z.abs_ci[0] < z.abs_lift < z.abs_ci[1])
    check("relative CI is asymmetric on the log scale",
          abs((z.rel_ci[1] - z.rel_lift) - (z.rel_lift - z.rel_ci[0])) > 1e-6)


def test_bayes_identities() -> None:
    """E[loss keep] - E[loss ship] must equal the posterior mean lift, exactly."""
    b = beta_binomial(1200, 10000, 1320, 10000)
    identity = b.expected_loss_keep - b.expected_loss_ship
    truth = b.post_mean_treatment - b.post_mean_control
    check("beta expected-loss identity", abs(identity - truth) < 1e-9,
          f"{identity} vs {truth}")
    # Large-sample sanity: P(T>C) should agree with the normal approximation.
    se = np.sqrt(0.12 * 0.88 / 10000 + 0.132 * 0.868 / 10000)
    approx = stats.norm.cdf((0.132 - 0.120) / se)
    check("beta P(T>C) matches normal approximation",
          abs(b.prob_treatment_better - approx) < 0.01,
          f"{b.prob_treatment_better} vs {approx}")
    check("credible interval excludes zero for a clear win",
          b.cred_interval_abs[0] > 0)

    rng = np.random.default_rng(1)
    y_c = rng.gamma(2.0, 1.0, 8000)
    y_t = rng.gamma(2.0, 1.05, 8000)
    n = normal_normal(y_c, y_t)
    identity = n.expected_loss_keep - n.expected_loss_ship
    truth = n.post_mean_treatment - n.post_mean_control
    check("normal expected-loss identity", abs(identity - truth) < 1e-9)
    check("normal credible interval contains the point estimate",
          n.cred_interval_abs[0] < truth < n.cred_interval_abs[1])


def test_bh_correction() -> None:
    """BH on a hand-checkable vector, plus the FDR guarantee under a pure null."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.5])
    rejected, adjusted = benjamini_hochberg(p, q=0.05)
    # q*i/m = .00714 .0143 .0214 .0286 .0357 .0429 .05 -> largest i with p_i<=thr is 2.
    check("BH rejects exactly the first two", rejected.tolist() ==
          [True, True, False, False, False, False, False], str(rejected))
    check("BH adjusted p monotone", np.all(np.diff(np.sort(adjusted)) >= -1e-12))
    check("BH adjusted p >= raw p", np.all(adjusted >= p - 1e-12))

    # Under a global null, BH's FDR collapses to P(any rejection) and must sit at q,
    # whereas 20 uncorrected tests at 0.05 fire ~64% of the time.
    rng = np.random.default_rng(3)
    n_runs = 1000
    bh_fires = uncorrected_fires = 0
    for _ in range(n_runs):
        draw = rng.uniform(size=20)
        rej, _ = benjamini_hochberg(draw, q=0.10)
        bh_fires += int(rej.any())
        uncorrected_fires += int((draw < 0.05).any())
    bh_rate, raw_rate = bh_fires / n_runs, uncorrected_fires / n_runs
    check("BH global-null error rate sits at q", 0.06 <= bh_rate <= 0.15,
          f"rate={bh_rate}")
    check("uncorrected scanning fires far more often", raw_rate > 4 * bh_rate,
          f"raw={raw_rate} bh={bh_rate}")


def test_obf_boundaries() -> None:
    """Reproduce published Lan-DeMets O'Brien-Fleming boundaries."""
    b = alpha_spending_boundaries(np.linspace(0.2, 1.0, 5), alpha=0.05, spending="obf")
    # Published Lan-DeMets OBF boundaries, 5 equally spaced looks, two-sided alpha=0.05.
    reference = np.array([4.8769, 3.3569, 2.6803, 2.2898, 2.0310])
    check("OBF 5-look boundaries match published values",
          np.max(np.abs(b.z_boundaries - reference)) < 0.01,
          str(np.round(b.z_boundaries, 4)))
    ten = alpha_spending_boundaries(np.linspace(0.1, 1.0, 10), alpha=0.05,
                                    spending="obf")
    reference10 = np.array([6.9914, 4.8768, 3.9188, 3.3565, 2.9852,
                            2.7100, 2.4933, 2.3313, 2.1928, 2.0742])
    check("OBF 10-look boundaries match published values",
          np.max(np.abs(ten.z_boundaries - reference10)) < 0.01,
          str(np.round(ten.z_boundaries, 4)))
    check("OBF boundaries strictly decrease", np.all(np.diff(b.z_boundaries) < 0))
    check("cumulative alpha ends at alpha", abs(b.alpha_spent[-1] - 0.05) < 1e-9)

    p = alpha_spending_boundaries(np.linspace(0.2, 1.0, 5), alpha=0.05,
                                  spending="pocock")
    check("Pocock boundaries are nearly flat",
          np.ptp(p.z_boundaries) < 0.35, str(np.round(p.z_boundaries, 3)))
    check("Pocock spends more early than OBF",
          p.alpha_spent[0] > b.alpha_spent[0] * 50)


def test_always_valid_monotone() -> None:
    z = np.array([[0.5, 1.9, 1.2, 3.4, 2.0]])
    se = np.full((1, 5), 0.003)
    p = always_valid_pvalues(z, se, tau=0.01)
    check("always-valid p is a running minimum", np.all(np.diff(p[0]) <= 1e-15))
    check("always-valid p is a probability", np.all((p >= 0) & (p <= 1)))
    # Larger |z| at the same information must not increase the p-value.
    p_big = always_valid_pvalues(np.array([[4.0]]), np.array([[0.003]]), tau=0.01)
    p_small = always_valid_pvalues(np.array([[1.0]]), np.array([[0.003]]), tau=0.01)
    check("always-valid p decreases in |z|", p_big[0, 0] < p_small[0, 0])


def test_peeking_fpr() -> None:
    """The headline claim: naive peeking inflates FPR, sequential methods do not."""
    sim = peeking_simulation(n_sims=2000, n_looks=10, n_per_arm_final=20000,
                             base_rate=0.10, true_rel_lift=0.0, alpha=0.05, seed=SEED)
    check("fixed-horizon FPR is nominal", 0.03 <= sim.fixed_horizon_rate <= 0.07,
          f"{sim.fixed_horizon_rate}")
    check("naive peeking FPR is grossly inflated", sim.naive_rate >= 0.12,
          f"{sim.naive_rate}")
    check("naive peeking is >2.5x the sequential methods",
          sim.naive_rate > 2.5 * max(sim.obf_rate, sim.msprt_rate),
          f"naive={sim.naive_rate} obf={sim.obf_rate} msprt={sim.msprt_rate}")
    check("alpha spending controls FPR", sim.obf_rate <= 0.07, f"{sim.obf_rate}")
    check("mixture SPRT controls FPR", sim.msprt_rate <= 0.05, f"{sim.msprt_rate}")

    power = peeking_simulation(n_sims=1500, n_looks=10, n_per_arm_final=20000,
                               base_rate=0.10, true_rel_lift=0.10, alpha=0.05,
                               seed=SEED + 1)
    check("alpha spending keeps most of the fixed-horizon power",
          power.obf_rate >= 0.85, f"{power.obf_rate}")
    check("mixture SPRT still has usable power", power.msprt_rate >= 0.55,
          f"{power.msprt_rate}")
    return sim


def test_power_roundtrip() -> None:
    """Sample size and MDE must be inverses, and the plan must deliver its power."""
    plan = sample_size_proportion(0.10, 0.05, alpha=0.05, power=0.80)
    mde = mde_proportion(0.10, plan.n_control, alpha=0.05, power=0.80)
    check("sample-size / MDE round trip", abs(mde - 0.05) < 5e-4, f"{mde}")
    achieved = achieved_power_proportion(0.10, 0.05, plan.n_control, alpha=0.05)
    check("planned design achieves 80% power", abs(achieved - 0.80) < 0.01,
          f"{achieved}")
    smaller = sample_size_proportion(0.10, 0.05, alpha=0.05, power=0.80,
                                     variance_reduction=0.23)
    check("variance reduction shrinks the required sample",
          abs(smaller.n_control / plan.n_control - 0.77) < 0.01,
          f"{smaller.n_control / plan.n_control}")
    unequal = sample_size_proportion(0.10, 0.05, alpha=0.05, power=0.80, ratio=0.2)
    check("a 1:5 split needs more total traffic than 1:1",
          unequal.n_total > plan.n_total * 1.5)


def test_srm_guard() -> None:
    ok = srm_check({"control": 50112, "treatment": 49888}, {"control": .5, "treatment": .5})
    check("balanced split passes SRM", ok.passed, f"p={ok.p_value}")
    bad = srm_check({"control": 48608, "treatment": 51392},
                    {"control": .5, "treatment": .5})
    check("1.5% skew at n=100k fails SRM", not bad.passed, f"p={bad.p_value}")
    check("SRM chi-square is large", bad.chi2 > 50, f"chi2={bad.chi2}")
    check("SRM names the offending arm", bad.worst_arm == "treatment")
    three = srm_check({"a": 3400, "b": 3300, "c": 3300},
                      {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    check("multi-arm SRM has dof = k-1", three.dof == 2)


def test_cuped() -> None:
    df, _ = generate("winner", seed=SEED)
    c = df[df.variant == "control"]
    t = df[df.variant == "treatment"]
    res = apply_cuped(c.revenue.to_numpy(), c.pre_revenue.to_numpy(),
                      t.revenue.to_numpy(), t.pre_revenue.to_numpy())
    check("CUPED reduces variance materially", res.variance_reduction > 0.15,
          f"{res.variance_reduction}")
    check("variance reduction equals rho^2",
          abs(res.variance_reduction - res.correlation**2) < 1e-9,
          f"{res.variance_reduction} vs {res.correlation**2}")
    check("effective sample size grows", res.ess_multiplier > 1.15,
          f"{res.ess_multiplier}")
    check("CUPED shrinks the SE of the lift", res.se_adjusted < res.se_raw)
    check("CUPED does not move the point estimate much",
          abs(res.adjusted_test.abs_lift - res.raw_test.abs_lift)
          < 0.35 * res.se_raw,
          f"{res.adjusted_test.abs_lift} vs {res.raw_test.abs_lift}")
    check("pre-period covariate is balanced across arms",
          res.balance_p_value > 0.01, f"{res.balance_p_value}")
    return res


def test_determinism() -> None:
    a, _ = generate("winner", seed=SEED)
    b, _ = generate("winner", seed=SEED)
    check("generator is deterministic",
          a["converted"].sum() == b["converted"].sum()
          and abs(a["revenue"].sum() - b["revenue"].sum()) < 1e-9)
    r1 = beta_binomial(1200, 10000, 1320, 10000)
    r2 = beta_binomial(1200, 10000, 1320, 10000)
    check("Bayesian layer is deterministic",
          r1.prob_treatment_better == r2.prob_treatment_better)


def test_end_to_end() -> dict:
    """Ground truth from the generator must survive the whole pipeline."""
    config = DecisionConfig()
    out = {}
    for case in ("null", "winner", "srm"):
        df, spec = generate(case, seed=SEED)
        out[case] = analyze(df, name=case, designed_split=spec.designed_split,
                            config=config, dimensions=SEGMENT_DIMENSIONS)

    null = out["null"]
    check("null: SRM passes", null.srm.passed)
    check("null: not shipped", null.recommendation.verdict != "SHIP",
          null.recommendation.verdict)
    check("null: fixed-horizon p is not significant", null.conversion.p_value > 0.05,
          f"{null.conversion.p_value}")
    check("null: always-valid p never crossed",
          float(null.monitor.always_valid_p[-1]) > 0.05,
          f"{null.monitor.always_valid_p[-1]}")
    check("null: CI contains zero",
          null.conversion.abs_ci[0] < 0 < null.conversion.abs_ci[1])
    check("null: BH kills every segment discovery",
          null.segments.n_significant_bh == 0,
          f"{null.segments.n_significant_bh}")
    check("null: at least one segment looked significant before correction",
          null.segments.n_significant_uncorrected >= 1,
          f"{null.segments.n_significant_uncorrected}")

    win = out["winner"]
    check("winner: SRM passes", win.srm.passed)
    check("winner: shipped", win.recommendation.verdict == "SHIP",
          win.recommendation.verdict)
    check("winner: effect is strongly significant", win.conversion.p_value < 1e-4,
          f"{win.conversion.p_value}")
    check("winner: always-valid p also crosses",
          float(win.monitor.always_valid_p[-1]) < 0.05,
          f"{win.monitor.always_valid_p[-1]}")
    check("winner: relative lift in the expected range",
          0.05 < win.conversion.rel_lift < 0.11, f"{win.conversion.rel_lift}")
    check("winner: CI excludes zero", win.conversion.abs_ci[0] > 0)
    check("winner: posterior is decisive",
          win.bayes_conversion.prob_treatment_better > 0.999,
          f"{win.bayes_conversion.prob_treatment_better}")
    check("winner: expected loss from shipping is negligible",
          win.bayes_conversion.expected_loss_ship_rel < 1e-3,
          f"{win.bayes_conversion.expected_loss_ship_rel}")
    check("winner: the true mobile interaction is discovered",
          "mobile" in set(win.segments.discoveries["level"]),
          str(win.segments.discoveries[["dimension", "level"]].to_dict("records")))
    check("winner: mobile lift exceeds the non-mobile lift",
          float(win.segments.frame.query("dimension=='device' and level=='mobile'")
                ["abs_lift"].iloc[0]) >
          float(win.segments.frame.query("dimension=='device' and level=='mobile'")
                ["lift_outside"].iloc[0]))

    srm = out["srm"]
    check("srm: guard fires", not srm.srm.passed, f"p={srm.srm.p_value}")
    check("srm: p-value is decisive", srm.srm.p_value < 1e-6, f"{srm.srm.p_value}")
    check("srm: analysis is blocked even though the lift is real",
          srm.recommendation.verdict == "BLOCKED - SRM", srm.recommendation.verdict)
    check("srm: the underlying lift really was positive",
          srm.conversion.rel_lift > 0.03, f"{srm.conversion.rel_lift}")

    df_win, spec_win = generate("winner", seed=SEED)
    early = analyze(truncate_to_day(df_win, 3), name="early",
                    designed_split=spec_win.designed_split, config=config,
                    planned_n_per_arm=len(df_win) // 2)
    check("early look is not called", early.recommendation.verdict == "KEEP RUNNING",
          early.recommendation.verdict)
    check("early look has not crossed the always-valid boundary",
          float(early.monitor.always_valid_p[-1]) > 0.05,
          f"{early.monitor.always_valid_p[-1]}")
    out["early"] = early
    return out


def main() -> int:
    t0 = time.perf_counter()
    test_frequentist_matches_scipy()
    test_bayes_identities()
    test_bh_correction()
    test_obf_boundaries()
    test_always_valid_monotone()
    sim = test_peeking_fpr()
    test_power_roundtrip()
    test_srm_guard()
    cuped = test_cuped()
    test_determinism()
    res = test_end_to_end()
    elapsed = time.perf_counter() - t0
    print(
        "OK - "
        f"peeking FPR naive={sim.naive_rate:.3f} vs OBF={sim.obf_rate:.3f} "
        f"mSPRT={sim.msprt_rate:.3f} (nominal 0.050); "
        f"CUPED var reduction={cuped.variance_reduction * 100:.1f}% "
        f"(ESS x{cuped.ess_multiplier:.2f}); "
        f"null verdict={res['null'].recommendation.verdict} "
        f"(p={res['null'].conversion.p_value:.3f}); "
        f"winner verdict={res['winner'].recommendation.verdict} "
        f"(lift={res['winner'].conversion.rel_lift * 100:+.2f}%, "
        f"p={res['winner'].conversion.p_value:.2e}); "
        f"srm verdict={res['srm'].recommendation.verdict} "
        f"(chi2={res['srm'].srm.chi2:.1f}); "
        f"HTE BH discoveries null={res['null'].segments.n_significant_bh} "
        f"winner={res['winner'].segments.n_significant_bh}; "
        f"{elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
