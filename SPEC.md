# Censorship Resistance Simulation Spec

## Goal
Compare inclusion behavior for:
- non-committee sequencing
- committee-based sequencing
- committee-based sequencing with the Aztec Alpha escape hatch fallback

under either a static sequencer set or configurable user-operated sequencer growth.

## Model (Current App Behavior)
- Initial network: `base_sequencers` total, with censoring share `censor_fraction` fixed from genesis.
- User onboarding: `max_new_sequencers_per_epoch` honest sequencers become active each epoch (`epoch_slots` slots). If it is `0`, the sequencer set stays fixed for the full simulation.
- Added-sequencer mix: malicious additions are derived from honest additions using `honest_add_success_rate` (in `(0, 1]`), so lower success rates increase censoring share over time.
- Non-committee mode: slot proposer is drawn from all active sequencers.
- Committee mode: committee of size `committee_size` is sampled per epoch from a validator set lagged by `validator_set_lag_epochs`, and reused for that epoch's slots.
- Committee gate: inclusion is allowed only when committee censors `<= floor((committee_size - 1) / 3)`.
- Escape hatch fallback: assume the Alpha upgrade is active. The censored user/group already holds `1` bonded escape-hatch candidate slot before censorship begins, and `escape_hatch_other_candidates` models the other bonded slots. A hatch opens every `112` epochs for `2` epochs; if the user's slot is designated proposer for that hatch, they can bypass the committee during the open window.
- Escape hatch bond: the user's `332,000,000` token bond stays active until the slot is selected or voluntarily exited, so the user does not pay the withdrawal tax before inclusion. The `1,660,000` token withdrawal tax is only paid on exit.
- Other escape-hatch candidates: any other candidate selected for a hatch is removed from the active set. To keep the pool crowded after being selected, that candidate must later exit and rejoin, paying the withdrawal tax again. Under the model's one-user-slot assumption, the coalition's expected tax burn until the user is selected is `escape_hatch_other_candidates * withdrawal_tax`.
- Combined committee + escape hatch output: overall survival is the product of committee-path survival and escape-hatch survival, i.e. the two fallback paths are treated as independent randomness sources.
- Time horizon: `horizon_slots = floor(max_horizon_days * 24 * 3600 / slot_seconds)`.

## Reference Committee Baseline
For the stationary committee-only baseline, the cleanest reference is a hypergeometric model over committee composition. This omits sequencer growth and the escape hatch, so it is a baseline for the committee path rather than the full app output.
- Runnable reference script: `python3 scripts/reference_committee_baseline.py`
- Defaults match the app's current static preset; edit the constants at the top if needed

## Inputs
- `base_sequencers`
- `stake_per_sequencer_token`
- `token_usd`
- `censor_fraction`
- `committee_size`
- `slot_seconds`
- `max_horizon_days`
- `target_inclusion_percent`
- `epoch_slots`
- `validator_set_lag_epochs`
- `max_new_sequencers_per_epoch`
- `honest_add_success_rate`
- `escape_hatch_other_candidates`

Validation:
- `committee_size` must be `<= base_sequencers` at simulation start.
- `target_inclusion_percent` must be in `(0, 100)`.
- `escape_hatch_other_candidates` must be in `[0, 8]`, based on the website assumption of `~3B` circulating AZTEC and at most one user/group slot.

## Outputs
- Cumulative inclusion chart over elapsed days for the full `max_horizon_days`.
- Effective per-slot inclusion chart: uses full `max_horizon_days`.
- Expected inclusion delay chart: uses full `max_horizon_days`; y-axis shown only up to 720 hours (30 days). This chart excludes the escape hatch and remains a sequencer-stake-only view.
- Target cards show time to reach the configured cumulative inclusion probability for committee, committee + escape hatch, and non-committee modes.

## UI Actions
- `Run Simulation`: validate inputs, recompute data, rerender charts.
- `Reset Defaults`: restore defaults from `DEFAULT_CONFIG` and rerun.
- URL sync: on successful run, current non-default config values are written to query params; valid query params are loaded on page open.

## Run
- Open `web/index.html` directly, or serve the `web/` directory with any static server.
- Plotly is loaded from CDN (`cdn.plot.ly`), so network access is required for charts.
