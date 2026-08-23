"""A/B Experiment Analyzer — Task Runner.

Run named tasks:
    python tasks.py demo
    python tasks.py peek
    python tasks.py plan
"""
from __future__ import annotations

import sys
from pathlib import Path

from abx.datagen import generate
from abx.decide import DecisionConfig
from abx.hte import scan_segments
from abx.pipeline import analyze, write_artifacts
from abx.power import sample_size_proportion
from abx.sequential import peeking_simulation

OUT = Path(__file__).parent / "output"


def task_demo() -> None:
    OUT.mkdir(exist_ok=True)
    print("Running A/B Experiment Analysis Pipeline...")
    df, _ = generate("winner", seed=42)
    config = DecisionConfig()
    res = analyze(df, name="winner_demo", config=config)
    print(res.report())
    write_artifacts(res, OUT)
    print(f"Artifacts saved -> {OUT}")


def task_peek() -> None:
    sim = peeking_simulation(n_sims=300, n_looks=5, n_per_arm_final=10000, base_rate=0.10, true_rel_lift=0.0, seed=42)
    print(sim.table())


def task_plan() -> None:
    plan = sample_size_proportion(0.10, 0.05, alpha=0.05, power=0.80)
    print(plan.table())


TASKS = {
    "demo": task_demo,
    "peek": task_peek,
    "plan": task_plan,
}

if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if task_name in TASKS:
        TASKS[task_name]()
    else:
        print(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
