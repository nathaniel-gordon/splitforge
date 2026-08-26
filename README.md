# SplitForge — Statistical A/B Experiment Engine

> Rigorous experiment analysis without guesswork. SplitForge automates the full statistical lifecycle of A/B tests: hypothesis validation, sample-size projection, sequential testing, frequentist and Bayesian significance, and multi-metric dashboards — so product teams get answers, not spreadsheets.

## What SplitForge Does

- **Pre-experiment planning** — MDE-based sample size calculator with power curves
- **Sequential testing** — always-valid p-values via mSPRT; stop early with confidence
- **Dual inference** — frequentist t-test / chi-square *and* Bayesian Beta-Binomial side by side
- **Multi-metric reporting** — revenue, conversion, engagement in a single experiment report
- **Segment drilldown** — detect HTE (heterogeneous treatment effects) per cohort

## Architecture

```
Experiment Config
    └─> SplitForge Planner  (power analysis, randomization check)
    └─> StatEngine          (t-test, Mann-Whitney, Beta-Binomial)
    └─> SequentialGuard     (mSPRT boundaries, early-stop logic)
    └─> ReportBuilder       (Markdown + CSV experiment summary)
```

## Quickstart

```bash
python walkthrough.py        # full end-to-end experiment walkthrough
python tasks.py demo         # quick stats engine demo
```

## Test

```bash
python tests/test_smoke.py
```

---

## 👤 Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
