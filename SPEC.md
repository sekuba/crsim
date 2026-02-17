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
- Central control: `horizon_days`.
- Derived slots:
  - `horizon_slots = floor(horizon_days * 24 * 3600 / slot_seconds)`.

## Outputs
- `results/cr_simulation.csv`
- `figures/cr_probability_vs_stake.svg`
- `figures/cr_per_slot_probability_vs_stake.svg`
- `figures/cr_expected_delay_vs_stake.svg`

## Config Reference (`sim/cr_simulation.py`)

- `base_sequencers`: initial sequencer count in the network.
- `stake_per_sequencer_token`: stake required per sequencer identity.
- `token_usd`: token USD price for x-axis conversion.
- `censor_fraction`: fraction of initial network that censors.
- `committee_size`: committee size `k`.
- `slot_seconds`: slot duration.
- `horizon_days`: analysis horizon in days (main control knob).
- `epoch_slots`: slots per epoch (onboarding cadence).
- `max_new_sequencers_per_epoch`: user onboarding throughput.
- `probability_near_one_margin`: x-axis auto-crop threshold for probability chart; crop begins once both curves are within this margin of 1.

Delay chart scaling:
- x/y bounds are auto-derived from generated data (finite delay values), with small padding only when a bound would collapse to a single point.
- y-axis is capped at 30 days (720 hours); autoscaling is based on values at or below the cap, and values above the cap are clipped in the plot.

## Run
```bash
python sim/cr_simulation.py
```
