"use strict";

const EPSILON = 1e-15;
const DELAY_Y_CAP_HOURS = 30 * 24;
const COMMITTEE_COLOR = "#2563EB";
const COMMITTEE_EH_COLOR = "#059669";
const NON_COMMITTEE_COLOR = "#F97316";

const EH_BOND_TOKENS = 332000000;
const EH_WITHDRAWAL_TAX_TOKENS = 1660000;
const EH_FREQUENCY_EPOCHS = 112;
const EH_ACTIVE_DURATION_EPOCHS = 2;
const EH_USER_CANDIDATE_SLOTS = 1;
const EH_CIRCULATING_SUPPLY_TOKENS = 3000000000;
const EH_MAX_OTHER_CANDIDATES =
  Math.floor(EH_CIRCULATING_SUPPLY_TOKENS / EH_BOND_TOKENS) - EH_USER_CANDIDATE_SLOTS;

// Example defaults aligned to the Aztec Alpha upgrade.
const DEFAULT_CONFIG = {
  base_sequencers: 4000,
  stake_per_sequencer_token: 200000,
  token_usd: 0.02,
  censor_fraction: 0.5,
  committee_size: 48,
  slot_seconds: 72,
  max_horizon_days: 10,
  target_inclusion_percent: 90,
  epoch_slots: 32,
  validator_set_lag_epochs: 2,
  max_new_sequencers_per_epoch: 4,
  honest_add_success_rate: 0.6,
  escape_hatch_other_candidates: 1,
};

const INTEGER_FIELDS = new Set([
  "base_sequencers",
  "stake_per_sequencer_token",
  "committee_size",
  "slot_seconds",
  "epoch_slots",
  "validator_set_lag_epochs",
  "max_new_sequencers_per_epoch",
  "escape_hatch_other_candidates",
]);

const formEl = document.getElementById("config-form");
const statusEl = document.getElementById("status");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");
const targetCommitteeLabelEl = document.getElementById("target-committee-label");
const targetCommitteeEhLabelEl = document.getElementById("target-committee-eh-label");
const targetNonLabelEl = document.getElementById("target-non-label");
const targetCommitteeValueEl = document.getElementById("target-committee-value");
const targetCommitteeEhValueEl = document.getElementById("target-committee-eh-value");
const targetNonValueEl = document.getElementById("target-non-value");
const targetCommitteeCostEl = document.getElementById("target-committee-cost");
const targetCommitteeEhCostEl = document.getElementById("target-committee-eh-cost");
const targetNonCostEl = document.getElementById("target-non-cost");
const targetHelpEl = document.getElementById("target-help");
const escapeHatchSummaryEl = document.getElementById("escape-hatch-summary");
const cumulativeChartEl = document.getElementById("chart-cumulative");
const perSlotChartEl = document.getElementById("chart-per-slot");
const delayChartEl = document.getElementById("chart-delay");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b42318" : "#334155";
}

function formatPercentValue(value) {
  return Number(value.toFixed(6)).toString();
}

function targetLabel(percent) {
  return `T${formatPercentValue(percent)}`;
}

function targetPercentText(percent) {
  return `${formatPercentValue(percent)}%`;
}

function syncTargetLabels(cfg) {
  const label = targetLabel(cfg.target_inclusion_percent);
  const percentText = targetPercentText(cfg.target_inclusion_percent);
  targetCommitteeLabelEl.textContent = `Committee ${label}`;
  targetCommitteeEhLabelEl.textContent = `Committee + EH ${label}`;
  targetNonLabelEl.textContent = `Non-committee ${label}`;
  targetHelpEl.textContent =
    `Time until cumulative inclusion probability reaches ${percentText} for the current scenario. `
    + "Committee + EH adds one pre-positioned user/group escape-hatch candidate slot to the ordinary committee path.";
}

function formatHumanDurationDays(days) {
  const totalSeconds = days * 24 * 3600;
  if (totalSeconds >= 48 * 3600) {
    return `${days.toFixed(2)}d`;
  }

  const totalHours = totalSeconds / 3600;
  if (totalHours >= 1) {
    return `${totalHours.toFixed(2)}h`;
  }

  const totalMinutes = totalSeconds / 60;
  if (totalMinutes >= 1) {
    return `${totalMinutes.toFixed(0)}m`;
  }

  return `${totalSeconds.toFixed(0)}s`;
}

function targetCardText(days, maxHorizonDays) {
  if (days === null) {
    return `>${formatHumanDurationDays(maxHorizonDays)}`;
  }
  return formatHumanDurationDays(days);
}

function formatWholeNumber(value) {
  return Math.round(value).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
}

function targetCostText(usd, token, label) {
  if (usd === null || token === null) {
    return `Cost @${label}: n/a`;
  }
  return `Cost @${label}: $${formatWholeNumber(usd)} | ${formatWholeNumber(token)} tok`;
}

function targetCommitteeEhText(stakeUsd, stakeToken, ehBondUsd, ehTaxUsd, label) {
  const stakeText = stakeUsd === null || stakeToken === null
    ? `stake @${label}: n/a`
    : `stake @${label}: $${formatWholeNumber(stakeUsd)} | ${formatWholeNumber(stakeToken)} tok`;
  return `${stakeText} | EH lock: $${formatWholeNumber(ehBondUsd)} | exit tax later: $${formatWholeNumber(ehTaxUsd)}`;
}

function clampProbability(value) {
  if (value <= 0.0) {
    return 0.0;
  }
  if (value >= 1.0) {
    return 1.0;
  }
  return value;
}

function maxCensorsAllowed(committeeSize) {
  return Math.floor((committeeSize - 1) / 3);
}

function logGamma(z) {
  const p = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019571e-6,
    1.5056327351493116e-7,
  ];

  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }

  let x = 0.9999999999998099;
  const y = z - 1;
  for (let i = 0; i < p.length; i += 1) {
    x += p[i] / (y + i + 1);
  }
  const t = y + p.length - 0.5;
  return 0.9189385332046727 + (y + 0.5) * Math.log(t) - t + Math.log(x);
}

function logChoose(n, k) {
  if (k < 0 || k > n) {
    return Number.NEGATIVE_INFINITY;
  }
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1);
}

function geometricSum(r, n) {
  if (n <= 0) {
    return 0.0;
  }
  if (Math.abs(1.0 - r) < EPSILON) {
    return n;
  }
  return (1.0 - r ** n) / (1.0 - r);
}

function paddedBounds(values, relativePadding = 0.05, absolutePadding = 1.0) {
  const lower = Math.min(...values);
  const upper = Math.max(...values);
  if (upper - lower > EPSILON) {
    return [lower, upper];
  }
  const pad = Math.max(Math.abs(lower) * relativePadding, absolutePadding);
  return [lower - pad, lower + pad];
}

function formatDurationHours(hours) {
  if (!Number.isFinite(hours)) {
    return "n/a";
  }
  if (hours >= 48) {
    return `${(hours / 24).toFixed(2)}d`;
  }
  return `${hours.toFixed(2)}h`;
}

function validateConfig(cfg) {
  const errors = [];
  if (cfg.base_sequencers < 0) {
    errors.push("base_sequencers must be >= 0");
  }
  if (cfg.stake_per_sequencer_token <= 0) {
    errors.push("stake_per_sequencer_token must be > 0");
  }
  if (cfg.token_usd <= 0) {
    errors.push("token_usd must be > 0");
  }
  if (cfg.censor_fraction < 0 || cfg.censor_fraction > 1) {
    errors.push("censor_fraction must be in [0, 1]");
  }
  if (cfg.committee_size <= 0) {
    errors.push("committee_size must be > 0");
  }
  if (cfg.committee_size > cfg.base_sequencers) {
    errors.push("committee_size must be <= base_sequencers at simulation start");
  }
  if (cfg.slot_seconds <= 0) {
    errors.push("slot_seconds must be > 0");
  }
  if (cfg.max_horizon_days <= 0) {
    errors.push("max_horizon_days must be > 0");
  }
  if (cfg.target_inclusion_percent <= 0 || cfg.target_inclusion_percent >= 100) {
    errors.push("target_inclusion_percent must be in (0, 100)");
  }
  if (cfg.epoch_slots <= 0) {
    errors.push("epoch_slots must be > 0");
  }
  if (cfg.validator_set_lag_epochs < 0) {
    errors.push("validator_set_lag_epochs must be >= 0");
  }
  if (cfg.max_new_sequencers_per_epoch < 0) {
    errors.push("max_new_sequencers_per_epoch must be >= 0");
  }
  if (cfg.honest_add_success_rate <= 0 || cfg.honest_add_success_rate > 1) {
    errors.push("honest_add_success_rate must be in (0, 1]");
  }
  if (
    cfg.escape_hatch_other_candidates < 0
    || cfg.escape_hatch_other_candidates > EH_MAX_OTHER_CANDIDATES
  ) {
    errors.push(`escape_hatch_other_candidates must be in [0, ${EH_MAX_OTHER_CANDIDATES}]`);
  }
  const derivedSlots = Math.floor((cfg.max_horizon_days * 24 * 3600) / cfg.slot_seconds);
  if (derivedSlots <= 0) {
    errors.push("max_horizon_days and slot_seconds imply 0 slots; increase max_horizon_days or decrease slot_seconds");
  }
  return errors;
}

function maxHorizonSlots(cfg) {
  return Math.floor((cfg.max_horizon_days * 24 * 3600) / cfg.slot_seconds);
}

function maxUserSequencers(cfg) {
  return Math.floor(maxHorizonSlots(cfg) / cfg.epoch_slots) * cfg.max_new_sequencers_per_epoch;
}

function maxHorizonUserSeqValues(cfg) {
  const maxSeq = maxUserSequencers(cfg);
  if (maxSeq <= 0) {
    return [0];
  }
  const step = Math.max(1, Math.floor(maxSeq / 250));
  const values = [];
  for (let x = 0; x <= maxSeq; x += step) {
    values.push(x);
  }
  if (values[values.length - 1] !== maxSeq) {
    values.push(maxSeq);
  }
  return values;
}

class SimulationModel {
  constructor(cfg) {
    this.cfg = cfg;
    const rawCensors = Math.round(cfg.base_sequencers * cfg.censor_fraction);
    this.initialCensors = Math.min(Math.max(rawCensors, 0), cfg.base_sequencers);
    this.initialHonest = cfg.base_sequencers - this.initialCensors;
    this.cache = new Map();
  }

  pNonCommitteeSlot(userSeq, maliciousSeq = 0) {
    const totalSeq = this.cfg.base_sequencers + userSeq + maliciousSeq;
    if (totalSeq <= 0) {
      return 0;
    }
    return clampProbability((userSeq + this.initialHonest) / totalSeq);
  }

  committeeDistribution(totalSeq, censorSeq, committeeSize) {
    if (committeeSize > totalSeq) {
      return [];
    }
    const honestSeq = totalSeq - censorSeq;
    const lo = Math.max(0, committeeSize - honestSeq);
    const hi = Math.min(committeeSize, censorSeq);
    if (lo > hi) {
      return [];
    }

    // Build probabilities relative to the mode to avoid underflow for large committees.
    const span = hi - lo + 1;
    const weights = new Array(span).fill(0.0);
    const modeRaw = Math.floor(((committeeSize + 1) * (censorSeq + 1)) / (totalSeq + 2));
    const mode = Math.min(hi, Math.max(lo, modeRaw));
    weights[mode - lo] = 1.0;

    for (let c = mode + 1; c <= hi; c += 1) {
      const prev = c - 1;
      const numer = (censorSeq - prev) * (committeeSize - prev);
      const denom = c * (honestSeq - committeeSize + c);
      weights[c - lo] = denom <= 0 ? 0.0 : weights[prev - lo] * (numer / denom);
    }

    for (let c = mode - 1; c >= lo; c -= 1) {
      const numer = (c + 1) * (honestSeq - committeeSize + c + 1);
      const denom = (censorSeq - c) * (committeeSize - c);
      weights[c - lo] = denom <= 0 ? 0.0 : weights[c + 1 - lo] * (numer / denom);
    }

    let sum = 0.0;
    for (const w of weights) {
      sum += w;
    }

    if (!Number.isFinite(sum) || sum <= EPSILON) {
      const logWeights = [];
      let maxLog = Number.NEGATIVE_INFINITY;
      for (let c = lo; c <= hi; c += 1) {
        const logP = logChoose(censorSeq, c)
          + logChoose(honestSeq, committeeSize - c)
          - logChoose(totalSeq, committeeSize);
        logWeights.push(logP);
        if (logP > maxLog) {
          maxLog = logP;
        }
      }
      sum = 0.0;
      for (let i = 0; i < logWeights.length; i += 1) {
        const scaled = Math.exp(logWeights[i] - maxLog);
        weights[i] = scaled;
        sum += scaled;
      }
      if (!Number.isFinite(sum) || sum <= EPSILON) {
        return [];
      }
    }

    const invSum = 1.0 / sum;
    const dist = [];
    for (let c = lo; c <= hi; c += 1) {
      dist.push([c, weights[c - lo] * invSum]);
    }
    return dist;
  }

  committeeEpochFactors(userSeq, maliciousSeq, slotsInEpoch) {
    const k = this.cfg.committee_size;
    const key = `${userSeq}|${maliciousSeq}|${slotsInEpoch}`;
    if (this.cache.has(key)) {
      return this.cache.get(key);
    }

    const totalSeq = this.cfg.base_sequencers + userSeq + maliciousSeq;
    const censorSeq = this.initialCensors + maliciousSeq;
    if (k > totalSeq) {
      const fallback = [1.0, slotsInEpoch];
      this.cache.set(key, fallback);
      return fallback;
    }

    const maxC = maxCensorsAllowed(k);
    let fTotal = 0.0;
    let gTotal = 0.0;
    let mass = 0.0;

    for (const [censorCommittee, p] of this.committeeDistribution(totalSeq, censorSeq, k)) {
      mass += p;
      const miss = censorCommittee / k;
      let f = 1.0;
      let g = slotsInEpoch;
      if (censorCommittee <= maxC) {
        f = miss ** slotsInEpoch;
        g = geometricSum(miss, slotsInEpoch);
      }
      fTotal += p * f;
      gTotal += p * g;
    }

    let result;
    if (mass <= EPSILON) {
      result = [1.0, slotsInEpoch];
    } else {
      const invMass = 1.0 / mass;
      result = [
        fTotal * invMass,
        gTotal * invMass,
      ];
    }

    this.cache.set(key, result);
    return result;
  }
}

function epochFactors(model, activeUser, activeMalicious, slotsInEpoch, committeeMode) {
  if (committeeMode) {
    const [f, g] = model.committeeEpochFactors(activeUser, activeMalicious, slotsInEpoch);
    return [f, g];
  }
  const pNon = model.pNonCommitteeSlot(activeUser, activeMalicious);
  const miss = 1.0 - pNon;
  return [miss ** slotsInEpoch, geometricSum(miss, slotsInEpoch)];
}

// Aztec committees sample from a validator set that is lagged by a fixed number of epochs.
function scheduledUserSequencersForEpoch(cfg, epochIdx, lagEpochs = 0) {
  const effectiveEpochIdx = Math.max(0, epochIdx - lagEpochs);
  return effectiveEpochIdx * cfg.max_new_sequencers_per_epoch;
}

function activeUserForEpoch(cfg, targetUserSeq, epochIdx, lagEpochs = 0) {
  return Math.min(targetUserSeq, scheduledUserSequencersForEpoch(cfg, epochIdx, lagEpochs));
}

function epochsToSteadyState(cfg, targetUserSeq, lagEpochs = 0) {
  if (targetUserSeq <= 0) {
    return 0;
  }
  if (cfg.max_new_sequencers_per_epoch <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.ceil(targetUserSeq / cfg.max_new_sequencers_per_epoch) + lagEpochs;
}

function maliciousSequencersForHonest(cfg, honestUserSeq) {
  if (honestUserSeq <= 0) {
    return 0;
  }
  const h = cfg.honest_add_success_rate;
  if (h >= 1.0 - EPSILON) {
    return 0;
  }
  return Math.max(0, Math.round((honestUserSeq * (1.0 - h)) / h));
}

function escapeHatchCycleSlots(cfg) {
  return EH_FREQUENCY_EPOCHS * cfg.epoch_slots;
}

function escapeHatchOpenSlots(cfg) {
  return Math.min(escapeHatchCycleSlots(cfg), EH_ACTIVE_DURATION_EPOCHS * cfg.epoch_slots);
}

function escapeHatchSelectionProbability(cfg) {
  return EH_USER_CANDIDATE_SLOTS / (EH_USER_CANDIDATE_SLOTS + cfg.escape_hatch_other_candidates);
}

function escapeHatchSurvivalProbability(cfg, elapsedSlots) {
  if (elapsedSlots <= 0) {
    return 1.0;
  }

  const cycleSlots = escapeHatchCycleSlots(cfg);
  const openSlots = escapeHatchOpenSlots(cfg);
  const miss = 1.0 - escapeHatchSelectionProbability(cfg);
  let totalSurvival = 0.0;

  // Assume censorship starts at a random point within the hatch cycle.
  // If the current hatch is open and the user's bonded slot was selected, they can force inclusion immediately.
  for (let phaseSlots = 0; phaseSlots < cycleSlots; phaseSlots += 1) {
    const attempts =
      (phaseSlots < openSlots ? 1 : 0)
      + Math.floor((elapsedSlots + phaseSlots) / cycleSlots);
    totalSurvival += miss ** attempts;
  }

  return clampProbability(totalSurvival / cycleSlots);
}

function escapeHatchExpectedDelayHours(cfg) {
  const cycleSlots = escapeHatchCycleSlots(cfg);
  const openSlots = escapeHatchOpenSlots(cfg);
  const selectionProbability = escapeHatchSelectionProbability(cfg);
  let totalDelaySlots = 0.0;

  for (let phaseSlots = 0; phaseSlots < cycleSlots; phaseSlots += 1) {
    const nextHatchDelay = (cycleSlots / selectionProbability) - phaseSlots;
    if (phaseSlots < openSlots) {
      totalDelaySlots += (1.0 - selectionProbability) * nextHatchDelay;
    } else {
      totalDelaySlots += nextHatchDelay;
    }
  }

  return (totalDelaySlots / cycleSlots) * (cfg.slot_seconds / 3600.0);
}

function escapeHatchExpectedOtherTaxToken(cfg) {
  return cfg.escape_hatch_other_candidates * EH_WITHDRAWAL_TAX_TOKENS;
}

function horizonProbabilityFromLogSurvival(logSurvival) {
  if (!Number.isFinite(logSurvival) && logSurvival < 0) {
    return 1.0;
  }
  const p = -Math.expm1(logSurvival);
  if (Math.abs(p) < EPSILON) {
    return 0.0;
  }
  return clampProbability(p);
}

function effectivePerSlotFromLogSurvival(logSurvival, slots) {
  if (slots <= 0) {
    return 0;
  }
  if (!Number.isFinite(logSurvival) && logSurvival < 0) {
    return 1.0;
  }
  const p = -Math.expm1(logSurvival / slots);
  if (Math.abs(p) < EPSILON) {
    return 0.0;
  }
  return clampProbability(p);
}

function expectedDelayHoursWithChurn(cfg, model, targetUserSeq, committeeMode) {
  const lagEpochs = committeeMode ? cfg.validator_set_lag_epochs : 0;
  const epochsToFull = epochsToSteadyState(cfg, targetUserSeq, lagEpochs);
  if (!Number.isFinite(epochsToFull)) {
    return Number.POSITIVE_INFINITY;
  }
  let survival = 1.0;
  let expectedSlots = 0.0;

  for (let epochIdx = 0; epochIdx < epochsToFull; epochIdx += 1) {
    const activeUser = activeUserForEpoch(cfg, targetUserSeq, epochIdx, lagEpochs);
    const activeMalicious = maliciousSequencersForHonest(cfg, activeUser);
    const [f, g] = epochFactors(
      model,
      activeUser,
      activeMalicious,
      cfg.epoch_slots,
      committeeMode
    );
    expectedSlots += survival * g;
    survival *= f;
  }

  const finalMalicious = maliciousSequencersForHonest(cfg, targetUserSeq);
  const [fFinal, gFinal] = epochFactors(
    model,
    targetUserSeq,
    finalMalicious,
    cfg.epoch_slots,
    committeeMode
  );
  if (fFinal >= 1.0 - EPSILON) {
    return Number.POSITIVE_INFINITY;
  }
  expectedSlots += (survival * gFinal) / (1.0 - fFinal);
  return expectedSlots * (cfg.slot_seconds / 3600.0);
}

function buildTimeSeries(cfg, model) {
  const maxSlots = maxHorizonSlots(cfg);
  const maxEpochs = Math.floor((maxSlots + cfg.epoch_slots - 1) / cfg.epoch_slots);
  const usdPerSeq = cfg.stake_per_sequencer_token * cfg.token_usd;

  const days = [0.0];
  const invested = [0.0];
  const userSeq = [0];
  const nonProbs = [0.0];
  const comProbs = [0.0];
  const comEhProbs = [0.0];
  const nonEffPerSlot = [0.0];
  const comEffPerSlot = [0.0];
  const comEhEffPerSlot = [0.0];

  let logSurvivalNon = 0.0;
  let logSurvivalCom = 0.0;
  let elapsedSlots = 0;

  for (let epochIdx = 0; epochIdx < maxEpochs; epochIdx += 1) {
    const slotsInEpoch = Math.min(cfg.epoch_slots, maxSlots - elapsedSlots);
    if (slotsInEpoch <= 0) {
      break;
    }
    const activeUserNon = scheduledUserSequencersForEpoch(cfg, epochIdx);
    const activeUserCom = scheduledUserSequencersForEpoch(cfg, epochIdx, cfg.validator_set_lag_epochs);
    const activeMaliciousNon = maliciousSequencersForHonest(cfg, activeUserNon);
    const activeMaliciousCom = maliciousSequencersForHonest(cfg, activeUserCom);
    const [fNon] = epochFactors(model, activeUserNon, activeMaliciousNon, slotsInEpoch, false);
    const [fCom] = epochFactors(model, activeUserCom, activeMaliciousCom, slotsInEpoch, true);

    if (fNon <= 0.0) {
      logSurvivalNon = Number.NEGATIVE_INFINITY;
    } else if (Number.isFinite(logSurvivalNon)) {
      logSurvivalNon += Math.log(fNon);
    }

    if (fCom <= 0.0) {
      logSurvivalCom = Number.NEGATIVE_INFINITY;
    } else if (Number.isFinite(logSurvivalCom)) {
      logSurvivalCom += Math.log(fCom);
    }

    elapsedSlots += slotsInEpoch;
    const currentUserSeq = activeUserNon;
    days.push((elapsedSlots * cfg.slot_seconds) / (24 * 3600));
    invested.push(currentUserSeq * usdPerSeq);
    userSeq.push(currentUserSeq);

    const ehSurvival = escapeHatchSurvivalProbability(cfg, elapsedSlots);
    const logSurvivalEh = ehSurvival <= 0.0 ? Number.NEGATIVE_INFINITY : Math.log(ehSurvival);
    const logSurvivalComEh =
      !Number.isFinite(logSurvivalCom) || !Number.isFinite(logSurvivalEh)
        ? Number.NEGATIVE_INFINITY
        : logSurvivalCom + logSurvivalEh;

    const pNon = horizonProbabilityFromLogSurvival(logSurvivalNon);
    const pCom = horizonProbabilityFromLogSurvival(logSurvivalCom);
    const pComEh = horizonProbabilityFromLogSurvival(logSurvivalComEh);
    nonProbs.push(pNon);
    comProbs.push(pCom);
    comEhProbs.push(pComEh);
    nonEffPerSlot.push(effectivePerSlotFromLogSurvival(logSurvivalNon, elapsedSlots));
    comEffPerSlot.push(effectivePerSlotFromLogSurvival(logSurvivalCom, elapsedSlots));
    comEhEffPerSlot.push(effectivePerSlotFromLogSurvival(logSurvivalComEh, elapsedSlots));
  }

  return {
    days,
    invested,
    userSeq,
    nonProbs,
    comProbs,
    comEhProbs,
    nonEffPerSlot,
    comEffPerSlot,
    comEhEffPerSlot,
  };
}

function firstIndexAtProbability(probabilities, threshold) {
  for (let i = 0; i < probabilities.length; i += 1) {
    if (probabilities[i] >= threshold) {
      return i;
    }
  }
  return null;
}

function firstDayAtProbability(days, probabilities, threshold) {
  const idx = firstIndexAtProbability(probabilities, threshold);
  if (idx === null || idx >= days.length) {
    return null;
  }
  return days[idx];
}

function valueAtIndexOrNull(values, idx) {
  if (idx === null || idx < 0 || idx >= values.length) {
    return null;
  }
  return values[idx];
}

function targetTokenCost(cfg, honestSeqCount) {
  if (honestSeqCount === null) {
    return null;
  }
  return honestSeqCount * cfg.stake_per_sequencer_token;
}

function targetUsdCost(cfg, tokenCost) {
  if (tokenCost === null) {
    return null;
  }
  return tokenCost * cfg.token_usd;
}

function runSimulation(cfg) {
  const usdPerSeq = cfg.stake_per_sequencer_token * cfg.token_usd;
  const model = new SimulationModel(cfg);
  const invested = [];
  const delayUserSeq = [];
  const nonDelay = [];
  const comDelay = [];
  const finiteDelayValues = [];

  // Sample the full horizon for the delay chart.
  for (const honestSeq of maxHorizonUserSeqValues(cfg)) {
    const investedUsd = honestSeq * usdPerSeq;
    const expectedHoursNon = expectedDelayHoursWithChurn(cfg, model, honestSeq, false);
    const expectedHoursCom = expectedDelayHoursWithChurn(cfg, model, honestSeq, true);

    invested.push(investedUsd);
    delayUserSeq.push(honestSeq);
    nonDelay.push(expectedHoursNon);
    comDelay.push(expectedHoursCom);

    if (Number.isFinite(expectedHoursNon)) {
      finiteDelayValues.push(expectedHoursNon);
    }
    if (Number.isFinite(expectedHoursCom)) {
      finiteDelayValues.push(expectedHoursCom);
    }
  }

  const visibleDelayValues = finiteDelayValues.filter((v) => v <= DELAY_Y_CAP_HOURS);
  const [delayXMin, delayXMax] = paddedBounds(invested, 0.05, usdPerSeq);

  let delayYMin = 0.0;
  let delayYMax = DELAY_Y_CAP_HOURS;
  if (visibleDelayValues.length) {
    [delayYMin, delayYMax] = paddedBounds(visibleDelayValues, 0.05, cfg.slot_seconds / 3600.0);
    delayYMax = Math.min(delayYMax, DELAY_Y_CAP_HOURS);
    if (delayYMax <= delayYMin) {
      delayYMin = 0.0;
      delayYMax = DELAY_Y_CAP_HOURS;
    }
  }

  const timeSeries = buildTimeSeries(cfg, model);
  const targetProbability = cfg.target_inclusion_percent / 100;
  const targetCommitteeIdx = firstIndexAtProbability(timeSeries.comProbs, targetProbability);
  const targetCommitteeEhIdx = firstIndexAtProbability(timeSeries.comEhProbs, targetProbability);
  const targetNonCommitteeIdx = firstIndexAtProbability(timeSeries.nonProbs, targetProbability);
  const targetCommitteeDays = firstDayAtProbability(timeSeries.days, timeSeries.comProbs, targetProbability);
  const targetCommitteeEhDays = firstDayAtProbability(timeSeries.days, timeSeries.comEhProbs, targetProbability);
  const targetNonCommitteeDays = firstDayAtProbability(timeSeries.days, timeSeries.nonProbs, targetProbability);
  const targetCommitteeHonestSeq = valueAtIndexOrNull(timeSeries.userSeq, targetCommitteeIdx);
  const targetCommitteeEhHonestSeq = valueAtIndexOrNull(timeSeries.userSeq, targetCommitteeEhIdx);
  const targetNonCommitteeHonestSeq = valueAtIndexOrNull(timeSeries.userSeq, targetNonCommitteeIdx);
  const targetCommitteeStakeToken = targetTokenCost(cfg, targetCommitteeHonestSeq);
  const targetCommitteeEhStakeToken = targetTokenCost(cfg, targetCommitteeEhHonestSeq);
  const targetNonCommitteeStakeToken = targetTokenCost(cfg, targetNonCommitteeHonestSeq);
  const targetCommitteeStakeUsd = targetUsdCost(cfg, targetCommitteeStakeToken);
  const targetCommitteeEhStakeUsd = targetUsdCost(cfg, targetCommitteeEhStakeToken);
  const targetNonCommitteeStakeUsd = targetUsdCost(cfg, targetNonCommitteeStakeToken);
  const ehBondUsd = EH_BOND_TOKENS * cfg.token_usd;
  const ehWithdrawalTaxUsd = EH_WITHDRAWAL_TAX_TOKENS * cfg.token_usd;
  const ehOtherBondToken = cfg.escape_hatch_other_candidates * EH_BOND_TOKENS;
  const ehOtherTaxToken = escapeHatchExpectedOtherTaxToken(cfg);

  return {
    meta: {
      usdPerSeq,
      slotSeconds: cfg.slot_seconds,
      epochSlots: cfg.epoch_slots,
      maxNewSequencersPerEpoch: cfg.max_new_sequencers_per_epoch,
      delayXMin,
      delayXMax,
      delayYMin,
      delayYMax,
      targetCommitteeDays,
      targetCommitteeEhDays,
      targetNonCommitteeDays,
      targetCommitteeStakeToken,
      targetCommitteeEhStakeToken,
      targetNonCommitteeStakeToken,
      targetCommitteeStakeUsd,
      targetCommitteeEhStakeUsd,
      targetNonCommitteeStakeUsd,
      escapeHatchOtherCandidates: cfg.escape_hatch_other_candidates,
      escapeHatchSelectionProbability: escapeHatchSelectionProbability(cfg),
      escapeHatchFrequencyDays: (escapeHatchCycleSlots(cfg) * cfg.slot_seconds) / (24 * 3600),
      escapeHatchActiveDurationHours: (escapeHatchOpenSlots(cfg) * cfg.slot_seconds) / 3600.0,
      escapeHatchExpectedDelayHours: escapeHatchExpectedDelayHours(cfg),
      escapeHatchBondToken: EH_BOND_TOKENS,
      escapeHatchBondUsd: ehBondUsd,
      escapeHatchWithdrawalTaxToken: EH_WITHDRAWAL_TAX_TOKENS,
      escapeHatchWithdrawalTaxUsd: ehWithdrawalTaxUsd,
      escapeHatchOtherBondToken: ehOtherBondToken,
      escapeHatchOtherBondUsd: ehOtherBondToken * cfg.token_usd,
      escapeHatchExpectedOtherTaxToken: ehOtherTaxToken,
      escapeHatchExpectedOtherTaxUsd: ehOtherTaxToken * cfg.token_usd,
    },
    series: {
      invested,
      userSeq: delayUserSeq,
      nonDelay,
      comDelay,
      cumulativeDays: timeSeries.days,
      cumulativeInvested: timeSeries.invested,
      cumulativeUserSeq: timeSeries.userSeq,
      cumulativeNonProbs: timeSeries.nonProbs,
      cumulativeComProbs: timeSeries.comProbs,
      cumulativeComEhProbs: timeSeries.comEhProbs,
      perSlotDays: timeSeries.days,
      perSlotInvested: timeSeries.invested,
      perSlotUserSeq: timeSeries.userSeq,
      perSlotNonEffPerSlot: timeSeries.nonEffPerSlot,
      perSlotComEffPerSlot: timeSeries.comEffPerSlot,
      perSlotComEhEffPerSlot: timeSeries.comEhEffPerSlot,
    },
  };
}

function formatDayTickLabel(days) {
  if (days >= 100) {
    return String(Math.round(days));
  }
  if (days >= 10) {
    return days.toFixed(1);
  }
  return days.toFixed(2);
}

function daysToReachUserSeq(userSeq, slotSeconds, epochSlots, maxNewPerEpoch) {
  if (userSeq <= 0) {
    return 0.0;
  }
  if (maxNewPerEpoch <= 0) {
    return Number.POSITIVE_INFINITY;
  }
  const epochs = Math.ceil(userSeq / maxNewPerEpoch);
  return (epochs * epochSlots * slotSeconds) / (24 * 3600);
}

function activeUserFromElapsedSlots(slots, epochSlots, maxNewPerEpoch) {
  if (slots <= 0) {
    return 0;
  }
  const epochIdx = Math.floor((slots - 1) / epochSlots);
  return Math.max(0, epochIdx * maxNewPerEpoch);
}

function buildTimeStakeTicks(xMinDays, xMaxDays, slotSeconds, epochSlots, maxNewPerEpoch, usdPerSeq) {
  const values = [];
  const labels = [];
  for (let i = 0; i <= 5; i += 1) {
    const dayValue = xMinDays + ((xMaxDays - xMinDays) * i) / 5;
    values.push(dayValue);

    const slots = Math.floor((dayValue * 24 * 3600) / slotSeconds);
    const seq = activeUserFromElapsedSlots(slots, epochSlots, maxNewPerEpoch);
    const usd = seq * usdPerSeq;
    labels.push(`${formatDayTickLabel(dayValue)}d | $${Math.round(usd).toLocaleString()} [${seq}]`);
  }
  return { values, labels };
}

function buildUsdSeqTimeTicks(xMinUsd, xMaxUsd, slotSeconds, epochSlots, maxNewPerEpoch, usdPerSeq) {
  const values = [];
  const labels = [];
  for (let i = 0; i <= 5; i += 1) {
    const usdValue = xMinUsd + ((xMaxUsd - xMinUsd) * i) / 5;
    values.push(usdValue);
    const seq = Math.max(0, Math.round(usdValue / usdPerSeq));
    const days = daysToReachUserSeq(seq, slotSeconds, epochSlots, maxNewPerEpoch);
    labels.push(`$${Math.round(usdValue).toLocaleString()} [${seq}] | ${formatDayTickLabel(days)}d`);
  }
  return { values, labels };
}

function toVisibleY(values, yMin, yMax) {
  return values.map((v) => {
    if (!Number.isFinite(v)) {
      return null;
    }
    if (v < yMin || v > yMax) {
      return null;
    }
    return v;
  });
}

function plotConfig() {
  return {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };
}

function baseLayout(xLabel, yLabel, xRange, yRange, xTicks) {
  return {
    margin: { l: 70, r: 30, t: 50, b: 60 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    legend: { orientation: "h", y: 1.12, x: 0 },
    xaxis: {
      title: xLabel,
      range: xRange,
      tickmode: "array",
      tickvals: xTicks.values,
      ticktext: xTicks.labels,
      gridcolor: "#eef2f7",
      zeroline: false,
    },
    yaxis: {
      title: yLabel,
      range: yRange,
      gridcolor: "#eef2f7",
      zeroline: false,
    },
  };
}

function renderCharts(output) {
  if (!window.Plotly) {
    throw new Error("Plotly failed to load. Check internet/CDN access.");
  }

  const cumulativeDays = output.series.cumulativeDays;
  const cumulativeTicks = buildTimeStakeTicks(
    Math.min(...cumulativeDays),
    Math.max(...cumulativeDays),
    output.meta.slotSeconds,
    output.meta.epochSlots,
    output.meta.maxNewSequencersPerEpoch,
    output.meta.usdPerSeq
  );
  const cumulativeHover = cumulativeDays.map((day, idx) => {
    const usd = Math.round(output.series.cumulativeInvested[idx]).toLocaleString();
    const seq = output.series.cumulativeUserSeq[idx].toLocaleString();
    return `Day ${day.toFixed(3)}<br>Invested: $${usd}<br>User sequencers: ${seq}`;
  });
  const cumulativeLayout = baseLayout(
    "Elapsed days | invested stake in USD [user sequencers]",
    "Cumulative inclusion probability",
    [Math.min(...cumulativeDays), Math.max(...cumulativeDays)],
    [0, 1],
    cumulativeTicks
  );
  const cumulativeData = [
    {
      x: cumulativeDays,
      y: output.series.cumulativeComEhProbs,
      mode: "lines",
      name: "Committee + EH",
      text: cumulativeHover,
      hovertemplate: "%{text}<br>Cumulative probability: %{y:.6f}<extra>Committee + EH</extra>",
      line: { color: COMMITTEE_EH_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: cumulativeDays,
      y: output.series.cumulativeComProbs,
      mode: "lines",
      name: "Committee",
      text: cumulativeHover,
      hovertemplate: "%{text}<br>Cumulative probability: %{y:.6f}<extra>Committee</extra>",
      line: { color: COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: cumulativeDays,
      y: output.series.cumulativeNonProbs,
      mode: "lines",
      name: "Non-committee",
      text: cumulativeHover,
      hovertemplate: "%{text}<br>Cumulative probability: %{y:.6f}<extra>Non-committee</extra>",
      line: { color: NON_COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
  ];

  const perSlotLayout = baseLayout(
    "Elapsed days | invested stake in USD [user sequencers]",
    "Effective per-slot inclusion probability",
    [Math.min(...output.series.perSlotDays), Math.max(...output.series.perSlotDays)],
    [0, 1],
    buildTimeStakeTicks(
      Math.min(...output.series.perSlotDays),
      Math.max(...output.series.perSlotDays),
      output.meta.slotSeconds,
      output.meta.epochSlots,
      output.meta.maxNewSequencersPerEpoch,
      output.meta.usdPerSeq
    )
  );
  const perSlotHover = output.series.perSlotDays.map((day, idx) => {
    const usd = Math.round(output.series.perSlotInvested[idx]).toLocaleString();
    const seq = output.series.perSlotUserSeq[idx].toLocaleString();
    return `Day ${day.toFixed(3)}<br>Invested: $${usd}<br>User sequencers: ${seq}`;
  });

  const perSlotData = [
    {
      x: output.series.perSlotDays,
      y: output.series.perSlotComEhEffPerSlot,
      mode: "lines",
      name: "Committee + EH",
      text: perSlotHover,
      hovertemplate: "%{text}<br>Effective per-slot: %{y:.6f}<extra>Committee + EH</extra>",
      line: { color: COMMITTEE_EH_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: output.series.perSlotDays,
      y: output.series.perSlotComEffPerSlot,
      mode: "lines",
      name: "Committee",
      text: perSlotHover,
      hovertemplate: "%{text}<br>Effective per-slot: %{y:.6f}<extra>Committee</extra>",
      line: { color: COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: output.series.perSlotDays,
      y: output.series.perSlotNonEffPerSlot,
      mode: "lines",
      name: "Non-committee",
      text: perSlotHover,
      hovertemplate: "%{text}<br>Effective per-slot: %{y:.6f}<extra>Non-committee</extra>",
      line: { color: NON_COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
  ];

  const invested = output.series.invested;

  const delayXTicks = buildUsdSeqTimeTicks(
    output.meta.delayXMin,
    output.meta.delayXMax,
    output.meta.slotSeconds,
    output.meta.epochSlots,
    output.meta.maxNewSequencersPerEpoch,
    output.meta.usdPerSeq
  );
  const delayYMin = output.meta.delayYMin;
  const delayYMax = output.meta.delayYMax;
  const delayHover = invested.map((usd, idx) => {
    const seq = output.series.userSeq[idx];
    const days = daysToReachUserSeq(
      seq,
      output.meta.slotSeconds,
      output.meta.epochSlots,
      output.meta.maxNewSequencersPerEpoch
    );
    return `Day ${days.toFixed(3)}<br>Invested: $${Math.round(usd).toLocaleString()}<br>User sequencers: ${seq.toLocaleString()}`;
  });

  const delayLayout = baseLayout(
    "Invested stake in USD [user sequencers]",
    "Expected time to inclusion (hours)",
    [output.meta.delayXMin, output.meta.delayXMax],
    [delayYMin, delayYMax],
    delayXTicks
  );

  const delayData = [
    {
      x: invested,
      y: toVisibleY(output.series.comDelay, delayYMin, delayYMax),
      mode: "lines",
      name: "Committee",
      text: delayHover,
      hovertemplate: "%{text}<br>Expected delay: %{y:.6f}h<extra>Committee</extra>",
      line: { color: COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: invested,
      y: toVisibleY(output.series.nonDelay, delayYMin, delayYMax),
      mode: "lines",
      name: "Non-committee",
      text: delayHover,
      hovertemplate: "%{text}<br>Expected delay: %{y:.6f}h<extra>Non-committee</extra>",
      line: { color: NON_COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
  ];

  Plotly.react(cumulativeChartEl, cumulativeData, cumulativeLayout, plotConfig());
  Plotly.react(perSlotChartEl, perSlotData, perSlotLayout, plotConfig());
  Plotly.react(delayChartEl, delayData, delayLayout, plotConfig());
}

function getConfigFromForm() {
  const formData = new FormData(formEl);
  const cfg = {};
  for (const [key, defaultValue] of Object.entries(DEFAULT_CONFIG)) {
    const parsed = Number(formData.get(key));
    cfg[key] = Number.isNaN(parsed) ? defaultValue : parsed;
    if (INTEGER_FIELDS.has(key)) {
      cfg[key] = Math.floor(cfg[key]);
    }
  }
  return cfg;
}

function setFormValues(values) {
  for (const [key, value] of Object.entries(values)) {
    const input = formEl.elements.namedItem(key);
    if (input) {
      input.value = String(value);
    }
  }
}

function configFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const cfg = { ...DEFAULT_CONFIG };
  let hasConfigParam = false;

  for (const key of Object.keys(DEFAULT_CONFIG)) {
    if (!params.has(key)) {
      continue;
    }
    const parsed = Number(params.get(key));
    if (!Number.isFinite(parsed)) {
      continue;
    }
    cfg[key] = INTEGER_FIELDS.has(key) ? Math.floor(parsed) : parsed;
    hasConfigParam = true;
  }

  return hasConfigParam ? cfg : null;
}

function configToUrl(cfg) {
  const params = new URLSearchParams(window.location.search);

  for (const [key, defaultValue] of Object.entries(DEFAULT_CONFIG)) {
    params.delete(key);
    if (cfg[key] !== defaultValue) {
      params.set(key, String(cfg[key]));
    }
  }

  const query = params.toString();
  const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
  window.history.replaceState({}, "", nextUrl);
}

function runFromForm() {
  setStatus("Running simulation...");
  const cfg = getConfigFromForm();
  syncTargetLabels(cfg);

  try {
    const errors = validateConfig(cfg);
    if (errors.length) {
      throw new Error(errors.join(" | "));
    }

    configToUrl(cfg);

    const output = runSimulation(cfg);
    renderCharts(output);

    const label = targetLabel(cfg.target_inclusion_percent);
    targetCommitteeValueEl.textContent = targetCardText(output.meta.targetCommitteeDays, cfg.max_horizon_days);
    targetCommitteeEhValueEl.textContent = targetCardText(output.meta.targetCommitteeEhDays, cfg.max_horizon_days);
    targetNonValueEl.textContent = targetCardText(output.meta.targetNonCommitteeDays, cfg.max_horizon_days);
    targetCommitteeCostEl.textContent = targetCostText(
      output.meta.targetCommitteeStakeUsd,
      output.meta.targetCommitteeStakeToken,
      label
    );
    targetCommitteeEhCostEl.textContent = targetCommitteeEhText(
      output.meta.targetCommitteeEhStakeUsd,
      output.meta.targetCommitteeEhStakeToken,
      output.meta.escapeHatchBondUsd,
      output.meta.escapeHatchWithdrawalTaxUsd,
      label
    );
    targetNonCostEl.textContent = targetCostText(
      output.meta.targetNonCommitteeStakeUsd,
      output.meta.targetNonCommitteeStakeToken,
      label
    );
    escapeHatchSummaryEl.textContent =
      `Escape hatch fallback assumes 1 pre-positioned user/group slot and `
      + `${output.meta.escapeHatchOtherCandidates} other bonded candidates. `
      + `Selection chance per hatch: ${(100 * output.meta.escapeHatchSelectionProbability).toFixed(2)}%. `
      + `A hatch opens every ${output.meta.escapeHatchFrequencyDays.toFixed(2)}d for `
      + `${formatDurationHours(output.meta.escapeHatchActiveDurationHours)}. `
      + `The user's bond stays active until selection or exit: `
      + `${formatWholeNumber(output.meta.escapeHatchBondToken)} tok `
      + `($${formatWholeNumber(output.meta.escapeHatchBondUsd)}) locked, with no EH tax paid before inclusion. `
      + `If the user later exits, the tax is ${formatWholeNumber(output.meta.escapeHatchWithdrawalTaxToken)} tok `
      + `($${formatWholeNumber(output.meta.escapeHatchWithdrawalTaxUsd)}). `
      + `Other EH participants keep ${formatWholeNumber(output.meta.escapeHatchOtherBondToken)} tok `
      + `($${formatWholeNumber(output.meta.escapeHatchOtherBondUsd)}) locked, and if they want to stay in the pool after being selected, `
      + `their coalition's expected tax burn until the user is selected is `
      + `${formatWholeNumber(output.meta.escapeHatchExpectedOtherTaxToken)} tok `
      + `($${formatWholeNumber(output.meta.escapeHatchExpectedOtherTaxUsd)}). `
      + `EH-only expected wait: ${formatDurationHours(output.meta.escapeHatchExpectedDelayHours)}.`;
    setStatus("Simulation complete.");
  } catch (error) {
    const label = targetLabel(cfg.target_inclusion_percent);
    targetCommitteeValueEl.textContent = "-";
    targetCommitteeEhValueEl.textContent = "-";
    targetNonValueEl.textContent = "-";
    targetCommitteeCostEl.textContent = `Cost @${label}: -`;
    targetCommitteeEhCostEl.textContent = `stake @${label}: -`;
    targetNonCostEl.textContent = `Cost @${label}: -`;
    escapeHatchSummaryEl.textContent = "-";
    setStatus(`Error: ${error.message}`, true);
  }
}

runBtn.addEventListener("click", runFromForm);
resetBtn.addEventListener("click", () => {
  setFormValues(DEFAULT_CONFIG);
  runFromForm();
});

const initialConfig = configFromUrl();
if (initialConfig && validateConfig(initialConfig).length === 0) {
  setFormValues(initialConfig);
} else {
  setFormValues(DEFAULT_CONFIG);
}
runFromForm();
