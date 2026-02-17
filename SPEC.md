# Censorship Resistance Simulation Spec

## Goal
Compare transaction inclusion chances for two L2 sequencing models as a function of user-operated sequencer stake.

## Scenarios

### 1) Non-committee model
- A slot has one proposer.
- If proposer is non-censoring, the censored user transaction is included.
- If proposer is censoring, the transaction is rejected.

Per-slot inclusion probability:
- `P(include) = P(non-censoring proposer)`

### 2) Committee model
- A slot has one proposer plus a committee.
- The committee is sampled once per epoch and reused for all slots in that epoch.
- In committee mode, the proposer for each slot is selected from that epoch's committee.
- Transaction is included only if:
1. proposer is non-censoring, and
2. committee passes the attestation condition.

Attestation condition used:
- censoring members in committee must be `< 1/3`
- with committee size `k=24`, this means censorers `<= 7` (equivalently at least 17 non-censoring members).

Epoch-level inclusion (for an epoch with `m` slots):
- Committee composition is random. Let `C` be censoring members in committee (`k` total members).
- Committee allows inclusion if `C <= floor((k-1)/3)` (equivalent to censorers `< 1/3`).
- If allowed, proposer is sampled from committee each slot, so per-slot non-censoring proposer probability is `(k - C) / k`.
- Conditional on `C`, epoch inclusion probability is:
  - `0`, if committee does not allow.
  - `1 - (1 - (k - C)/k)^m`, if committee allows.
- Overall epoch inclusion is the expectation over the committee composition distribution.

## Churn / Onboarding
- User can run multiple sequencers.
- User sequencers become active gradually:
  - max `max_new_sequencers_per_epoch` added per epoch,
  - epoch length is `epoch_slots` slots.
- Since committees are epoch-scoped, committee gate and user active set both update at epoch boundaries.
- The simulated stake range is inferred from horizon:
  - `max_user_sequencers = floor(horizon_slots / epoch_slots) * max_new_sequencers_per_epoch`

## Time Horizon
- Central control: `max_horizon_days`.
- Derived slots:
  - `horizon_slots = floor(max_horizon_days * 24 * 3600 / slot_seconds)`.
  - In the web app, cumulative plot can stop early once both modes reach `1 - probability_near_one_margin`.
  - Per-slot and delay plots keep the full `max_horizon_days` range.

## Outputs
- `results/cr_simulation.csv`
- `figures/cr_probability_vs_stake.svg`
- `figures/cr_per_slot_probability_vs_stake.svg`
- `figures/cr_expected_delay_vs_stake.svg`

## Config Reference (`web/app.js`)

- `base_sequencers`: initial sequencer count in the network.
- `stake_per_sequencer_token`: stake required per sequencer identity.
- `token_usd`: token USD price for x-axis conversion.
- `censor_fraction`: fraction of initial network that censors.
- `committee_size`: committee size `k`.
- `slot_seconds`: slot duration.
- `max_horizon_days`: maximum analysis horizon in days (upper bound for all web chart ranges; fractional values like `0.1` are allowed).
- `epoch_slots`: slots per epoch (onboarding cadence).
- `max_new_sequencers_per_epoch`: user onboarding throughput.
- `probability_near_one_margin`: stopping threshold for cumulative horizon search; cumulative plot stops once both curves are within this margin of 1 (unless `max_horizon_days` is hit first).

Delay chart scaling:
- x/y bounds are auto-derived from generated data (finite delay values), with small padding only when a bound would collapse to a single point.
- y-axis is capped at 30 days (720 hours); autoscaling is based on values at or below the cap, and values above the cap are clipped in the plot.

## Run
```bash
python sim/cr_simulation.py
```

## Static Web App
- Open `web/index.html` in a browser (or serve `web/` with any static server).
- Charts use Plotly for interactivity (zoom, pan, reset, legend toggles, image export).
- Includes a cumulative inclusion chart over elapsed time; x-ticks show derived invested USD and user sequencer count under the "only user onboarding" assumption.
- Effective per-slot chart uses elapsed-time x-ticks in the same format, but always spans full `max_horizon_days`.
- Delay chart x-ticks also show USD, user sequencer count, and derived onboarding time while spanning full `max_horizon_days`.
- Actions:
  - `Run Simulation`: recompute charts and in-memory table.
  - `Reset Defaults`: restore default config values.
  - `Download CSV`: export current results as `cr_simulation.csv`.
