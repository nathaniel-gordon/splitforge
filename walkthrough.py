"""A/B Experiment Analyzer — Narrative Walkthrough Script.

Run directly:
    python walkthrough.py

This script demonstrates model-view separation in statistical inference:
  - Part 1: Peeking simulation demonstrating alpha-inflation under sequential testing
  - Part 2: CUPED variance reduction and SRM (Sample Ratio Mismatch) diagnostic
  - Part 3: Heterogeneous treatment effect (HTE) scan with Benjamini-Hochberg FDR control
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abx.datagen import generate
from abx.decide import DecisionConfig
from abx.hte import scan_segments
from abx.pipeline import analyze, write_artifacts
from abx.sequential import peeking_simulation

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# 1. Peeking Simulation — Alpha Spending vs Naive Testing
# ════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("PART 1: Peeking Simulation — Alpha Spending vs Naive Fixed-Horizon")
print("=" * 80)

sim = peeking_simulation(
    n_sims=500,
    n_looks=5,
    n_per_arm_final=10000,
    base_rate=0.10,
    true_rel_lift=0.0,
    seed=42,
)
print("Simulation on A/A null test (nominal alpha = 0.05):")
print(sim.table())

# ════════════════════════════════════════════════════════════════════════
# 2. Case Analysis — SRM Checks & CUPED Variance Reduction
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2: Case Analysis — Winner Scenario with CUPED Adjustment")
print("=" * 80)

df_winner, _ = generate("winner", seed=42)
config = DecisionConfig()
res = analyze(df_winner, name="checkout_winner", config=config)
print(res.report())

# ════════════════════════════════════════════════════════════════════════
# 3. Heterogeneous Treatment Effect (HTE) Subgroup Scan
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 3: HTE Subgroup Scan with Benjamini-Hochberg FDR Control")
print("=" * 80)

dims = ["device", "country", "browser"]
valid_dims = [d for d in dims if d in df_winner.columns]
scan = scan_segments(df_winner, valid_dims, q=0.10)
print(f"Tested {scan.n_hypotheses} subgroup hypotheses:")
print(scan.table(top=6))

write_artifacts(res, OUT)
print(f"\nArtifacts saved to {OUT}")
