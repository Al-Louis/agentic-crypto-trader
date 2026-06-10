"""The ONE definition of the entry-forward residual reward.

Both `event_env` (training) and `scripts/preflight_entry_forward` (the landscape gate) import this
exact function — so the preflight can never again validate a different objective than we train on
(the exp3 false-PASS lesson: the preflight scored a global-demean proxy while the env implemented a
per-interval universe-demean, and the gate meant nothing).

The reward credits each ENTRY's deviation from the rung-0 rule by the chosen token's realized forward
return, demeaned by the *typical-ignition* return — which is precisely what the success gate
(`diagnostics.deviation_alpha`) measures (`dev = size − 0.20` vs the token's forward-24h return). So
the training objective and the success metric become the *same quantity*.
"""
from __future__ import annotations


def entry_forward_reward(dev_entry: float, fwd_ret: float, mu_base: float, res_gamma: float) -> float:
    """Per-entry residual: `dev·(fwd_ret − mu_base) − γ·dev²`.

    - `dev_entry`  = the agent's entry size − the rule's 0.20 (skip ⇒ −0.20).
    - `fwd_ret`    = the chosen token's realized return over the forward horizon (causal: paid only
      once that window has elapsed — see the env's semi-MDP maturation).
    - `mu_base`    = the typical-ignition forward return (the right null: over-sizing an *average*
      ignition nets 0, only *beating* the typical one pays → the all-big corner is genuinely zeroed).
    - `res_gamma`  = quadratic deviation budget → interior optimum `dev* = (fwd_ret−mu_base)/2γ`
      (rank-correct), so neither sizing corner is optimal.
    """
    return dev_entry * (fwd_ret - mu_base) - res_gamma * dev_entry * dev_entry
