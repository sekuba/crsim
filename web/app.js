"use strict";

const EPSILON = 1e-15;
const DELAY_Y_CAP_HOURS = 30 * 24;
const COMMITTEE_COLOR = "#2563EB";
const NON_COMMITTEE_COLOR = "#F97316";

// modeled after current aztec mainnet
const DEFAULT_CONFIG = {
  base_sequencers: 4000,
  stake_per_sequencer_token: 200000,
  token_usd: 0.02,
  censor_fraction: 0.67,
  committee_size: 24,
  slot_seconds: 72,
  max_horizon_days: 30,
  epoch_slots: 32,
  max_new_sequencers_per_epoch: 4,
  honest_add_success_rate: 0.5,
  probability_near_one_margin: 0.001,
};

const INTEGER_FIELDS = new Set([
  "base_sequencers",
  "stake_per_sequencer_token",
  "committee_size",
  "slot_seconds",
  "epoch_slots",
  "max_new_sequencers_per_epoch",
]);

const formEl = document.getElementById("config-form");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");
const csvBtn = document.getElementById("csv-btn");
const t90CommitteeValueEl = document.getElementById("t90-committee-value");
const t90NonValueEl = document.getElementById("t90-non-value");
const t90CommitteeCostEl = document.getElementById("t90-committee-cost");
const t90NonCostEl = document.getElementById("t90-non-cost");
const cumulativeChartEl = document.getElementById("chart-cumulative");
const perSlotChartEl = document.getElementById("chart-per-slot");
const delayChartEl = document.getElementById("chart-delay");

let latestCsv = "";

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b42318" : "#334155";
}

function t90CardText(days, maxHorizonDays) {
  if (days === null) {
    return `>${maxHorizonDays.toFixed(2)}d`;
  }
  return `${days.toFixed(2)}d`;
}

function formatWholeNumber(value) {
  return Math.round(value).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
}

function t90CostText(usd, token) {
  if (usd === null || token === null) {
    return "Cost @T90: n/a";
  }
  return `Cost @T90: $${formatWholeNumber(usd)} | ${formatWholeNumber(token)} tok`;
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
  if (cfg.epoch_slots <= 0) {
    errors.push("epoch_slots must be > 0");
  }
  if (cfg.max_new_sequencers_per_epoch <= 0) {
    errors.push("max_new_sequencers_per_epoch must be > 0");
  }
  if (cfg.honest_add_success_rate <= 0 || cfg.honest_add_success_rate > 1) {
    errors.push("honest_add_success_rate must be in (0, 1]");
  }
  if (cfg.probability_near_one_margin <= 0 || cfg.probability_near_one_margin >= 1) {
    errors.push("probability_near_one_margin must be in (0, 1)");
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

  committeeEpochFactors(userSeq, maliciousSeq, slotsInEpoch, committeeSize = null) {
    const k = committeeSize === null ? this.cfg.committee_size : committeeSize;
    const key = `${userSeq}|${maliciousSeq}|${slotsInEpoch}|${k}`;
    if (this.cache.has(key)) {
      return this.cache.get(key);
    }

    const totalSeq = this.cfg.base_sequencers + userSeq + maliciousSeq;
    const censorSeq = this.initialCensors + maliciousSeq;
    if (k > totalSeq) {
      const fallback = [1.0, slotsInEpoch, 0.0];
      this.cache.set(key, fallback);
      return fallback;
    }

    const maxC = maxCensorsAllowed(k);
    let pAllow = 0.0;
    let fTotal = 0.0;
    let gTotal = 0.0;
    let mass = 0.0;

    for (const [censorCommittee, p] of this.committeeDistribution(totalSeq, censorSeq, k)) {
      mass += p;
      const miss = censorCommittee / k;
      let f = 1.0;
      let g = slotsInEpoch;
      if (censorCommittee <= maxC) {
        pAllow += p;
        f = miss ** slotsInEpoch;
        g = geometricSum(miss, slotsInEpoch);
      }
      fTotal += p * f;
      gTotal += p * g;
    }

    let result;
    if (mass <= EPSILON) {
      result = [1.0, slotsInEpoch, 0.0];
    } else {
      const invMass = 1.0 / mass;
      result = [
        fTotal * invMass,
        gTotal * invMass,
        clampProbability(pAllow * invMass),
      ];
    }

    this.cache.set(key, result);
    return result;
  }
}

function epochFactors(model, activeUser, activeMalicious, slotsInEpoch, committeeMode, committeeSize = null) {
  if (committeeMode) {
    const [f, g] = model.committeeEpochFactors(activeUser, activeMalicious, slotsInEpoch, committeeSize);
    return [f, g];
  }
  const pNon = model.pNonCommitteeSlot(activeUser, activeMalicious);
  const miss = 1.0 - pNon;
  return [miss ** slotsInEpoch, geometricSum(miss, slotsInEpoch)];
}

function activeUserForEpoch(cfg, targetUserSeq, epochIdx) {
  return Math.min(targetUserSeq, epochIdx * cfg.max_new_sequencers_per_epoch);
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

function horizonLogSurvival(cfg, model, targetUserSeq, committeeMode, slots, committeeSize = null) {
  if (slots <= 0) {
    return 0.0;
  }
  let logSurvival = 0.0;
  const epochs = Math.floor((slots + cfg.epoch_slots - 1) / cfg.epoch_slots);
  for (let epochIdx = 0; epochIdx < epochs; epochIdx += 1) {
    const startSlot = epochIdx * cfg.epoch_slots;
    const slotsInEpoch = Math.min(cfg.epoch_slots, slots - startSlot);
    const activeUser = activeUserForEpoch(cfg, targetUserSeq, epochIdx);
    const activeMalicious = maliciousSequencersForHonest(cfg, activeUser);
    const [f] = epochFactors(
      model,
      activeUser,
      activeMalicious,
      slotsInEpoch,
      committeeMode,
      committeeSize
    );
    if (f <= 0) {
      return Number.NEGATIVE_INFINITY;
    }
    logSurvival += Math.log(f);
  }
  return logSurvival;
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

function expectedDelayHoursWithChurn(cfg, model, targetUserSeq, committeeMode, committeeSize = null) {
  const epochsToFull = Math.ceil(targetUserSeq / cfg.max_new_sequencers_per_epoch);
  let survival = 1.0;
  let expectedSlots = 0.0;

  for (let epochIdx = 0; epochIdx < epochsToFull; epochIdx += 1) {
    const activeUser = activeUserForEpoch(cfg, targetUserSeq, epochIdx);
    const activeMalicious = maliciousSequencersForHonest(cfg, activeUser);
    const [f, g] = epochFactors(
      model,
      activeUser,
      activeMalicious,
      cfg.epoch_slots,
      committeeMode,
      committeeSize
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
    committeeMode,
    committeeSize
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
  const nonEffPerSlot = [0.0];
  const comEffPerSlot = [0.0];

  let logSurvivalNon = 0.0;
  let logSurvivalCom = 0.0;
  let elapsedSlots = 0;

  for (let epochIdx = 0; epochIdx < maxEpochs; epochIdx += 1) {
    const slotsInEpoch = Math.min(cfg.epoch_slots, maxSlots - elapsedSlots);
    if (slotsInEpoch <= 0) {
      break;
    }
    const activeUser = epochIdx * cfg.max_new_sequencers_per_epoch;
    const activeMalicious = maliciousSequencersForHonest(cfg, activeUser);
    const [fNon] = epochFactors(model, activeUser, activeMalicious, slotsInEpoch, false);
    const [fCom] = epochFactors(model, activeUser, activeMalicious, slotsInEpoch, true);

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
    // Value at this time point reflects the sequencers active during the slots just simulated.
    const currentUserSeq = activeUser;
    days.push((elapsedSlots * cfg.slot_seconds) / (24 * 3600));
    invested.push(currentUserSeq * usdPerSeq);
    userSeq.push(currentUserSeq);

    const pNon = horizonProbabilityFromLogSurvival(logSurvivalNon);
    const pCom = horizonProbabilityFromLogSurvival(logSurvivalCom);
    nonProbs.push(pNon);
    comProbs.push(pCom);
    nonEffPerSlot.push(effectivePerSlotFromLogSurvival(logSurvivalNon, elapsedSlots));
    comEffPerSlot.push(effectivePerSlotFromLogSurvival(logSurvivalCom, elapsedSlots));
  }

  return {
    days,
    invested,
    userSeq,
    nonProbs,
    comProbs,
    nonEffPerSlot,
    comEffPerSlot,
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

function t90TokenCost(cfg, honestSeqCount) {
  if (honestSeqCount === null) {
    return null;
  }
  return honestSeqCount * cfg.stake_per_sequencer_token;
}

function t90UsdCost(cfg, tokenCost) {
  if (tokenCost === null) {
    return null;
  }
  return tokenCost * cfg.token_usd;
}

function runSimulation(cfg) {
  const slots = maxHorizonSlots(cfg);
  const usdPerSeq = cfg.stake_per_sequencer_token * cfg.token_usd;
  const horizonTag = `${cfg.max_horizon_days}d`;
  const pHorizonNonKey = `p_${horizonTag}_non_committee_with_churn_exact`;
  const pHorizonComKey = `p_${horizonTag}_committee_with_churn_exact`;
  const pEffNonKey = `p_eff_${horizonTag}_non_committee_per_slot`;
  const pEffComKey = `p_eff_${horizonTag}_committee_per_slot`;

  const model = new SimulationModel(cfg);
  const rows = [];

  // Keep row sampling on the full max-horizon range; cumulative charts may stop earlier.
  for (const userSeq of maxHorizonUserSeqValues(cfg)) {
    const maliciousSeq = maliciousSequencersForHonest(cfg, userSeq);
    const investedToken = userSeq * cfg.stake_per_sequencer_token;
    const investedUsd = investedToken * cfg.token_usd;
    const [, , pAllow] = model.committeeEpochFactors(userSeq, maliciousSeq, cfg.epoch_slots);

    const logSurvivalNon = horizonLogSurvival(cfg, model, userSeq, false, slots);
    const logSurvivalCom = horizonLogSurvival(cfg, model, userSeq, true, slots);

    rows.push({
      user_sequencers: userSeq,
      malicious_sequencers: maliciousSeq,
      invested_stake_token: investedToken,
      invested_stake_usd: investedUsd,
      p_committee_allows_honest: pAllow,
      [pHorizonNonKey]: horizonProbabilityFromLogSurvival(logSurvivalNon),
      [pHorizonComKey]: horizonProbabilityFromLogSurvival(logSurvivalCom),
      [pEffNonKey]: effectivePerSlotFromLogSurvival(logSurvivalNon, slots),
      [pEffComKey]: effectivePerSlotFromLogSurvival(logSurvivalCom, slots),
      expected_hours_non_committee: expectedDelayHoursWithChurn(cfg, model, userSeq, false),
      expected_hours_committee: expectedDelayHoursWithChurn(cfg, model, userSeq, true),
    });
  }

  const invested = rows.map((r) => Number(r.invested_stake_usd));
  const userSeq = rows.map((r) => Number(r.user_sequencers));
  const nonDelay = rows.map((r) => r.expected_hours_non_committee);
  const comDelay = rows.map((r) => r.expected_hours_committee);

  const finiteDelayValues = rows
    .flatMap((r) => [r.expected_hours_non_committee, r.expected_hours_committee])
    .filter((v) => Number.isFinite(v));

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
  const t90Target = 0.9;
  const t90CommitteeIdx = firstIndexAtProbability(timeSeries.comProbs, t90Target);
  const t90NonCommitteeIdx = firstIndexAtProbability(timeSeries.nonProbs, t90Target);
  const t90CommitteeDays = firstDayAtProbability(timeSeries.days, timeSeries.comProbs, t90Target);
  const t90NonCommitteeDays = firstDayAtProbability(timeSeries.days, timeSeries.nonProbs, t90Target);
  const t90CommitteeHonestSeq = valueAtIndexOrNull(timeSeries.userSeq, t90CommitteeIdx);
  const t90NonCommitteeHonestSeq = valueAtIndexOrNull(timeSeries.userSeq, t90NonCommitteeIdx);
  const t90CommitteeStakeToken = t90TokenCost(cfg, t90CommitteeHonestSeq);
  const t90NonCommitteeStakeToken = t90TokenCost(cfg, t90NonCommitteeHonestSeq);
  const t90CommitteeStakeUsd = t90UsdCost(cfg, t90CommitteeStakeToken);
  const t90NonCommitteeStakeUsd = t90UsdCost(cfg, t90NonCommitteeStakeToken);
  const targetProbability = 1.0 - cfg.probability_near_one_margin;
  let cumulativeEndIdx = timeSeries.days.length - 1;
  let cumulativeReachedTarget = false;
  for (let i = 0; i < timeSeries.days.length; i += 1) {
    if (timeSeries.nonProbs[i] >= targetProbability && timeSeries.comProbs[i] >= targetProbability) {
      cumulativeEndIdx = i;
      cumulativeReachedTarget = true;
      break;
    }
  }
  const toCumulativeSlice = (values) => values.slice(0, cumulativeEndIdx + 1);

  return {
    rows,
    meta: {
      maxHorizonSlots: slots,
      horizonTag,
      usdPerSeq,
      slotSeconds: cfg.slot_seconds,
      epochSlots: cfg.epoch_slots,
      maxNewSequencersPerEpoch: cfg.max_new_sequencers_per_epoch,
      maxUserSequencers: maxUserSequencers(cfg),
      userPointCount: rows.length,
      delayXMin,
      delayXMax,
      delayYMin,
      delayYMax,
      cumulativeReachedTarget,
      cumulativeTargetProbability: targetProbability,
      cumulativeXMaxDays: timeSeries.days[cumulativeEndIdx],
      cumulativeEndNonProb: timeSeries.nonProbs[cumulativeEndIdx],
      cumulativeEndComProb: timeSeries.comProbs[cumulativeEndIdx],
      t90TargetProbability: t90Target,
      t90CommitteeDays,
      t90NonCommitteeDays,
      t90CommitteeStakeToken,
      t90NonCommitteeStakeToken,
      t90CommitteeStakeUsd,
      t90NonCommitteeStakeUsd,
    },
    series: {
      invested,
      userSeq,
      nonDelay,
      comDelay,
      cumulativeDays: toCumulativeSlice(timeSeries.days),
      cumulativeInvested: toCumulativeSlice(timeSeries.invested),
      cumulativeUserSeq: toCumulativeSlice(timeSeries.userSeq),
      cumulativeNonProbs: toCumulativeSlice(timeSeries.nonProbs),
      cumulativeComProbs: toCumulativeSlice(timeSeries.comProbs),
      perSlotDays: timeSeries.days,
      perSlotInvested: timeSeries.invested,
      perSlotUserSeq: timeSeries.userSeq,
      perSlotNonEffPerSlot: timeSeries.nonEffPerSlot,
      perSlotComEffPerSlot: timeSeries.comEffPerSlot,
    },
  };
}

function buildUsdSeqTicks(xMin, xMax, usdPerSeq) {
  const values = [];
  const labels = [];
  for (let i = 0; i <= 5; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / 5;
    values.push(value);
    labels.push(`$${Math.round(value).toLocaleString()} [${Math.round(value / usdPerSeq)}]`);
  }
  return { values, labels };
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

function csvEscape(value) {
  let raw;
  if (value === Number.POSITIVE_INFINITY) {
    raw = "inf";
  } else if (value === Number.NEGATIVE_INFINITY) {
    raw = "-inf";
  } else {
    raw = String(value);
  }

  if (!raw.includes(",") && !raw.includes('"') && !raw.includes("\n")) {
    return raw;
  }
  return `"${raw.replaceAll('"', '""')}"`;
}

function rowsToCsv(rows) {
  if (!rows.length) {
    return "";
  }
  const keys = Object.keys(rows[0]);
  const lines = [keys.join(",")];
  for (const row of rows) {
    lines.push(keys.map((k) => csvEscape(row[k])).join(","));
  }
  return `${lines.join("\n")}\n`;
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

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function runFromForm() {
  setStatus("Running simulation...");
  summaryEl.textContent = "";

  try {
    const cfg = getConfigFromForm();
    const errors = validateConfig(cfg);
    if (errors.length) {
      throw new Error(errors.join(" | "));
    }

    configToUrl(cfg);

    const output = runSimulation(cfg);
    latestCsv = rowsToCsv(output.rows);

    renderCharts(output);

    t90CommitteeValueEl.textContent = t90CardText(output.meta.t90CommitteeDays, cfg.max_horizon_days);
    t90NonValueEl.textContent = t90CardText(output.meta.t90NonCommitteeDays, cfg.max_horizon_days);
    t90CommitteeCostEl.textContent = t90CostText(output.meta.t90CommitteeStakeUsd, output.meta.t90CommitteeStakeToken);
    t90NonCostEl.textContent = t90CostText(output.meta.t90NonCommitteeStakeUsd, output.meta.t90NonCommitteeStakeToken);

    const t90Label = `T${Math.round(output.meta.t90TargetProbability * 100)}`;
    const formatT90 = (days) => {
      if (days === null) {
        return `not reached by ${cfg.max_horizon_days.toFixed(2)} days`;
      }
      return `${days.toFixed(2)} days`;
    };
    const t90Summary = `${t90Label}: committee ${formatT90(output.meta.t90CommitteeDays)}, non-committee ${formatT90(output.meta.t90NonCommitteeDays)}`;
    const cumulativeTargetPct = (output.meta.cumulativeTargetProbability * 100).toFixed(3);
    const cumulativeSummary = output.meta.cumulativeReachedTarget
      ? `Cumulative >= ${cumulativeTargetPct}% by ${output.meta.cumulativeXMaxDays.toFixed(2)} days`
      : `Cumulative at ${output.meta.cumulativeXMaxDays.toFixed(2)} days: committee ${(output.meta.cumulativeEndComProb * 100).toFixed(3)}%, non-committee ${(output.meta.cumulativeEndNonProb * 100).toFixed(3)}%`;
    const assumptionsSummary = "Assumptions: onboarding updates at epoch boundaries; initial censor share is fixed from genesis; malicious additions are derived from honest_add_success_rate.";
    summaryEl.textContent = `Max horizon slots: ${output.meta.maxHorizonSlots.toLocaleString()} | Max user sequencers: ${output.meta.maxUserSequencers.toLocaleString()} | Stake samples: ${output.meta.userPointCount.toLocaleString()} | ${t90Summary} | ${cumulativeSummary} | ${assumptionsSummary}`;
    setStatus("Simulation complete.");
  } catch (error) {
    t90CommitteeValueEl.textContent = "-";
    t90NonValueEl.textContent = "-";
    t90CommitteeCostEl.textContent = "Cost @T90: -";
    t90NonCostEl.textContent = "Cost @T90: -";
    setStatus(`Error: ${error.message}`, true);
  }
}

runBtn.addEventListener("click", runFromForm);
resetBtn.addEventListener("click", () => {
  setFormValues(DEFAULT_CONFIG);
  runFromForm();
});
csvBtn.addEventListener("click", () => {
  if (!latestCsv) {
    runFromForm();
  }
  if (latestCsv) {
    downloadTextFile("cr_simulation.csv", latestCsv);
  }
});

const initialConfig = configFromUrl();
if (initialConfig && validateConfig(initialConfig).length === 0) {
  setFormValues(initialConfig);
} else {
  setFormValues(DEFAULT_CONFIG);
}
runFromForm();
