"""Seeded synthetic experiment generator with a KNOWN ground truth.

The generator exists so every claim the platform makes can be checked against the
truth that produced the data.  Three canonical cases ship with the demo:

  * "null"   - the treatment does literally nothing.  A correct platform must not
               declare a winner, and the sequential machinery must not be fooled by
               daily peeking.
  * "winner" - a genuine ~9% relative lift on conversion, concentrated on mobile so
               that the segment scan has a real interaction to discover.
  * "srm"    - a genuine lift AND a broken 51.5/48.5 bucketing bug.  A correct
               platform must refuse to report the (real!) lift, because the same bug
               that skewed the split could equally have skewed the metric.

Structure of the population.  Every user carries a latent quality score q ~ N(0,1)
that drives BOTH their pre-period behaviour and their in-experiment behaviour.  That
single shared factor is what makes the pre-period covariate predictive and therefore
what makes CUPED work - exactly the mechanism CUPED exploits on real traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEVICES = ("mobile", "desktop", "tablet")
DEVICE_P = (0.60, 0.30, 0.10)
DEVICE_LOGIT = {"mobile": -0.10, "desktop": 0.25, "tablet": 0.00}

COUNTRIES = ("US", "UK", "DE", "IN")
COUNTRY_P = (0.50, 0.20, 0.20, 0.10)
COUNTRY_LOGIT = {"US": 0.15, "UK": 0.05, "DE": 0.00, "IN": -0.30}

TENURES = ("new", "returning", "loyal")
TENURE_P = (0.40, 0.40, 0.20)
TENURE_LOGIT = {"new": -0.35, "returning": 0.05, "loyal": 0.45}

CHANNELS = ("organic", "paid", "referral")
CHANNEL_P = (0.45, 0.35, 0.20)
CHANNEL_LOGIT = {"organic": 0.10, "paid": -0.15, "referral": 0.05}

SEGMENT_DIMENSIONS = ["device", "country", "tenure", "channel"]


@dataclass
class ExperimentSpec:
    """Ground truth for one synthetic experiment."""

    name: str
    n_users: int
    designed_split: dict[str, float]
    realised_treatment_share: float
    base_logit: float
    treat_logit: float                      # uniform treatment shift on the logit
    treat_logit_by_device: dict[str, float] = field(default_factory=dict)
    revenue_treat_log: float = 0.0          # multiplicative revenue effect (log scale)
    n_days: int = 14
    description: str = ""

    @property
    def is_null(self) -> bool:
        return (self.treat_logit == 0.0
                and not any(self.treat_logit_by_device.values())
                and self.revenue_treat_log == 0.0)

    @property
    def has_srm(self) -> bool:
        designed = (self.designed_split["treatment"]
                    / sum(self.designed_split.values()))
        return abs(self.realised_treatment_share - designed) > 1e-9


CASES: dict[str, ExperimentSpec] = {
    "null": ExperimentSpec(
        name="null",
        n_users=100_000,
        designed_split={"control": 0.5, "treatment": 0.5},
        realised_treatment_share=0.5,
        base_logit=-2.35,
        treat_logit=0.0,
        description="A/A-style test: the treatment has no effect on any metric.",
    ),
    "winner": ExperimentSpec(
        name="winner",
        n_users=100_000,
        designed_split={"control": 0.5, "treatment": 0.5},
        realised_treatment_share=0.5,
        base_logit=-2.35,
        treat_logit=-0.03,
        treat_logit_by_device={"mobile": 0.22, "desktop": 0.0, "tablet": 0.0},
        revenue_treat_log=0.035,
        description=("Genuine win driven almost entirely by mobile; a real "
                     "interaction the segment scan should surface."),
    ),
    "srm": ExperimentSpec(
        name="srm",
        n_users=100_000,
        designed_split={"control": 0.5, "treatment": 0.5},
        realised_treatment_share=0.515,
        base_logit=-2.35,
        treat_logit=0.05,
        treat_logit_by_device={"mobile": 0.10, "desktop": 0.0, "tablet": 0.0},
        revenue_treat_log=0.03,
        description=("Bucketing bug: 51.5% of traffic lands in treatment. The lift is "
                     "real in the generator but unreportable in practice."),
    ),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _map_logit(values: np.ndarray, table: dict[str, float]) -> np.ndarray:
    out = np.zeros(values.size)
    for key, val in table.items():
        out[values == key] = val
    return out


def generate(case: str = "winner", seed: int = 42,
             n_users: int | None = None) -> tuple[pd.DataFrame, ExperimentSpec]:
    """Generate one experiment's user-level log.

    Columns: user_id, day, variant, device, country, tenure, channel,
             pre_revenue, converted, revenue.
    Everything is drawn vectorised - no Python loop touches a user row.
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose from {sorted(CASES)}")
    spec = CASES[case]
    if n_users is not None:
        spec = ExperimentSpec(**{**spec.__dict__, "n_users": int(n_users)})
    rng = np.random.default_rng(seed)
    n = spec.n_users

    q = rng.standard_normal(n)
    device = rng.choice(DEVICES, size=n, p=DEVICE_P)
    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_P)
    tenure = rng.choice(TENURES, size=n, p=TENURE_P)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_P)
    day = rng.integers(1, spec.n_days + 1, size=n)

    variant = np.where(rng.random(n) < spec.realised_treatment_share,
                       "treatment", "control")
    is_t = variant == "treatment"

    # Pre-period: same latent q, so the covariate is genuinely predictive.
    pre_active = rng.random(n) < _sigmoid(-0.45 + 1.20 * q)
    pre_amount = np.exp(1.55 + 0.80 * q + 0.45 * rng.standard_normal(n))
    pre_revenue = np.where(pre_active, pre_amount, 0.0)

    base = (spec.base_logit + 1.20 * q
            + _map_logit(device, DEVICE_LOGIT)
            + _map_logit(country, COUNTRY_LOGIT)
            + _map_logit(tenure, TENURE_LOGIT)
            + _map_logit(channel, CHANNEL_LOGIT))
    treat_shift = spec.treat_logit + _map_logit(device, spec.treat_logit_by_device)
    logit = base + np.where(is_t, treat_shift, 0.0)
    converted = rng.random(n) < _sigmoid(logit)

    amount = np.exp(1.55 + 0.80 * q + 0.45 * rng.standard_normal(n)
                    + np.where(is_t, spec.revenue_treat_log, 0.0))
    revenue = np.where(converted, amount, 0.0)

    df = pd.DataFrame({
        "user_id": np.arange(n, dtype=np.int64),
        "day": day.astype(np.int16),
        "variant": variant,
        "device": device,
        "country": country,
        "tenure": tenure,
        "channel": channel,
        "pre_revenue": pre_revenue,
        "converted": converted.astype(np.int8),
        "revenue": revenue,
    })
    return df, spec


def daily_cumulative(df: pd.DataFrame, control: str = "control",
                     treatment: str = "treatment") -> pd.DataFrame:
    """Cumulative per-arm exposures and conversions by day (the interim-look table)."""
    grouped = (df.groupby(["day", "variant"], observed=True)
                 .agg(n=("user_id", "size"), x=("converted", "sum"))
                 .unstack("variant", fill_value=0)
                 .sort_index())
    out = pd.DataFrame({
        "day": grouped.index.to_numpy(),
        "n_control": grouped[("n", control)].cumsum().to_numpy(),
        "x_control": grouped[("x", control)].cumsum().to_numpy(),
        "n_treatment": grouped[("n", treatment)].cumsum().to_numpy(),
        "x_treatment": grouped[("x", treatment)].cumsum().to_numpy(),
    })
    return out
