"""abx - rigorous A/B experiment analysis platform.

Modules are deliberately small and statistically self-contained so each piece can
be audited on its own: the frequentist tests, the Bayesian posteriors, the
always-valid sequential machinery, the SRM guard, CUPED, power planning and the
segment scan never share hidden state.
"""

__version__ = "1.0.0"
