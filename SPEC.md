# Censorship Resistance Simulation Spec

## Goal
Compare inclusion behavior for:
- non-committee sequencing
- committee-based sequencing

as user-operated sequencers are added over time.

## Model (Current App Behavior)
- Initial network: `base_sequencers` total, with censoring share `censor_fraction` fixed from genesis.
- User onboarding: `max_new_sequencers_per_epoch` honest sequencers become active each epoch (`epoch_slots` slots). If it is `0`, the sequencer set stays fixed for the full simulation.
- Added-sequencer mix: malicious additions are derived from honest additions using `honest_add_success_rate` (in `(0, 1]`), so lower success rates increase censoring share over time.
- Non-committee mode: slot proposer is drawn from all active sequencers.
- Committee mode: committee of size `committee_size` is sampled per epoch from a validator set lagged by `validator_set_lag_epochs`, and reused for that epoch's slots.
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
- `validator_set_lag_epochs`
- `max_new_sequencers_per_epoch`
- `honest_add_success_rate`

Validation:
- `committee_size` must be `<= base_sequencers` at simulation start.

## Outputs
- Cumulative inclusion chart over elapsed days for the full `max_horizon_days`.
- Effective per-slot inclusion chart: uses full `max_horizon_days`.
- Expected inclusion delay chart: uses full `max_horizon_days`; y-axis shown only up to 720 hours (30 days).
- `T90` cards show time to reach 90% cumulative inclusion probability for committee and non-committee modes.

## UI Actions
- `Run Simulation`: validate inputs, recompute data, rerender charts.
- `Reset Defaults`: restore defaults from `DEFAULT_CONFIG` and rerun.
- URL sync: on successful run, current non-default config values are written to query params; valid query params are loaded on page open.

## Run
- Open `web/index.html` directly, or serve the `web/` directory with any static server.
- Plotly is loaded from CDN (`cdn.plot.ly`), so network access is required for charts.
- Run statistical/regression checks with `node scripts/test-sim.js`.
