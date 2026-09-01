# SplitForge — Statistical A/B Testing Engine with Sequential mSPRT & Dual Inference

SplitForge is a statistical experimentation platform that eliminates common A/B testing pitfalls (peeking bias, underpowered tests, sample ratio mismatches). It provides simultaneous **Frequentist and Bayesian dual inference** alongside **mixture Sequential Probability Ratio Tests (mSPRT)** for continuous experiment monitoring.

## Statistical Methodologies

1. **Always-Valid Sequential Testing (mSPRT)**: Enables real-time stopping rules without inflating Type I error rates ($p < 0.05$ always valid).
2. **CUPED Variance Reduction**: Uses pre-experiment covariate data to reduce metric variance by up to $40\%$, cutting required sample sizes almost in half.
3. **Frequentist + Bayesian Dual View**: Side-by-side $p$-values and posterior distributions $P(	ext{Variant} > 	ext{Control} \mid 	ext{Data})$ for intuitive decision-making.

## Usage

```bash
# Run full experimentation walkthrough and Monte Carlo calibration
python walkthrough.py
```

## Tests

```bash
pytest tests/ -v
```
