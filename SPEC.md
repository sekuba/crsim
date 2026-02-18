# Censorship Resistance Simulation Spec

## Goal
Compare inclusion behavior for:
- non-committee sequencing
- committee-based sequencing

as user-operated sequencers are added over time.

## Model (Current App Behavior)
- Initial network: `base_sequencers` total, with censoring share `censor_fraction` fixed from genesis.
- User onboarding: up to `max_new_sequencers_per_epoch` sequencers become active each epoch (`epoch_slots` slots).
- Added-sequencer mix: malicious additions are derived from honest additions using `honest_add_success_rate` (in `(0, 1]`), so lower success rates increase censoring share over time.
- Non-committee mode: slot proposer is drawn from all active sequencers.
- Committee mode: committee of size `committee_size` is sampled per epoch and reused for that epoch's slots.
- Committee gate: inclusion is allowed only when committee censors `<= floor((committee_size - 1) / 3)`.
- Time horizon: `horizon_slots = floor(max_horizon_days * 24 * 3600 / slot_seconds)`.

## Inputs
- `base_sequencers`
- `stake_per_sequencer_token`
- `token_usd`
- `censor_fraction`
- `committee_size`
- `slot_seconds`
- `max_horizon_days`
- `epoch_slots`
- `max_new_sequencers_per_epoch`
- `honest_add_success_rate`
- `probability_near_one_margin`

## Outputs
- Cumulative inclusion chart over elapsed days: stops early once both modes reach `1 - probability_near_one_margin`, else uses full horizon.
- Effective per-slot inclusion chart: uses full `max_horizon_days`.
- Expected inclusion delay chart: uses full `max_horizon_days`; y-axis shown only up to 720 hours (30 days).
- CSV download (`cr_simulation.csv`) for sampled user-sequencer points, including invested stake, horizon probabilities, effective per-slot probabilities, and expected delays.
- Summary includes `T90` (time to reach 90% cumulative inclusion probability) for committee and non-committee modes.

## UI Actions
- `Run Simulation`: validate inputs, recompute data, rerender charts.
- `Reset Defaults`: restore defaults from `DEFAULT_CONFIG` and rerun.
- `Download CSV`: export current in-memory results.
- URL sync: on successful run, current non-default config values are written to query params; valid query params are loaded on page open.

## Run
- Open `web/index.html` directly, or serve the `web/` directory with any static server.
- Plotly is loaded from CDN (`cdn.plot.ly`), so network access is required for charts.
