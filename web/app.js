"use strict";

const EPSILON = 1e-15;
const COMMITTEE_COLOR = "#2563EB";
const COMMITTEE_EH_COLOR = "#059669";
const ETHEREUM_COLOR = "#F97316";
const REFERENCE_MARKER_COLOR = "#DC2626";

const EH_BOND_TOKENS = 332000000;
const EH_WITHDRAWAL_TAX_TOKENS = 1660000;
const EH_FREQUENCY_EPOCHS = 112;
const EH_ACTIVE_DURATION_EPOCHS = 2;
const EH_USER_CANDIDATE_SLOTS = 1;
const EH_CIRCULATING_SUPPLY_TOKENS = 3000000000;
const EH_MAX_OTHER_CANDIDATES =
  Math.floor(EH_CIRCULATING_SUPPLY_TOKENS / EH_BOND_TOKENS) - EH_USER_CANDIDATE_SLOTS;

// Default baseline: static sequencer set, 50% censoring, 99% inclusion target.
const DEFAULT_CONFIG = {
  base_sequencers: 4000,
  stake_per_sequencer_token: 200000,
  token_usd: 0.02,
  censor_fraction: 0.50,
  committee_size: 48,
  slot_seconds: 72,
  max_horizon_days: 20,
  target_inclusion_percent: 99,
  epoch_slots: 32,
  validator_set_lag_epochs: 2,
  max_new_sequencers_per_epoch: 0,
  honest_add_success_rate: 0.50,
  escape_hatch_other_candidates: 1,
};

// Single-proposer inclusion models can swap this slot-probability rule without touching chart code.
const ETHEREUM_INCLUSION_MODEL = Object.freeze({
  id: "ethereum",
  label: "Ethereum",
  slotProbability(simulationModel, userSeq, maliciousSeq = 0) {
    return simulationModel.pEthereumCanonicalSlot(userSeq, maliciousSeq);
  },
});

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
const targetEthereumLabelEl = document.getElementById("target-ethereum-label");
const targetCommitteeValueEl = document.getElementById("target-committee-value");
const targetCommitteeEhValueEl = document.getElementById("target-committee-eh-value");
const targetEthereumValueEl = document.getElementById("target-ethereum-value");
const targetCommitteeCostEl = document.getElementById("target-committee-cost");
const targetCommitteeEhCostEl = document.getElementById("target-committee-eh-cost");
const targetEthereumCostEl = document.getElementById("target-ethereum-cost");
const cumulativeChartEl = document.getElementById("chart-cumulative");
const perSlotChartEl = document.getElementById("chart-per-slot");
const referenceBaselineChartEl = document.getElementById("chart-reference-baseline");

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

function syncTargetLabels(cfg) {
  const label = targetLabel(cfg.target_inclusion_percent);
  targetCommitteeLabelEl.textContent = `Committee ${label}`;
  targetCommitteeEhLabelEl.textContent = `Committee + EH ${label}`;
  targetEthereumLabelEl.textContent = `${ETHEREUM_INCLUSION_MODEL.label} ${label}`;
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
  return `${stakeText} | EH lock: $${formatWholeNumber(ehBondUsd)} | exit tax: $${formatWholeNumber(ehTaxUsd)}`;
}

function renderTargetCards(meta, maxHorizonDays, label) {
  targetCommitteeValueEl.textContent = targetCardText(meta.targetCommitteeDays, maxHorizonDays);
  targetCommitteeEhValueEl.textContent = targetCardText(meta.targetCommitteeEhDays, maxHorizonDays);
  targetEthereumValueEl.textContent = targetCardText(meta.targetEthereumDays, maxHorizonDays);
  targetCommitteeCostEl.textContent = targetCostText(
    meta.targetCommitteeStakeUsd,
    meta.targetCommitteeStakeToken,
    label
  );
  targetCommitteeEhCostEl.textContent = targetCommitteeEhText(
    meta.targetCommitteeEhStakeUsd,
    meta.targetCommitteeEhStakeToken,
    meta.escapeHatchBondUsd,
    meta.escapeHatchWithdrawalTaxUsd,
    label
  );
  targetEthereumCostEl.textContent = targetCostText(
    meta.targetEthereumStakeUsd,
    meta.targetEthereumStakeToken,
    label
  );
}

function clearTargetCards(label) {
  targetCommitteeValueEl.textContent = "-";
  targetCommitteeEhValueEl.textContent = "-";
  targetEthereumValueEl.textContent = "-";
  targetCommitteeCostEl.textContent = `Cost @${label}: -`;
  targetCommitteeEhCostEl.textContent = `stake @${label}: -`;
  targetEthereumCostEl.textContent = `Cost @${label}: -`;
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

function logMultiplySurvival(logSurvival, factor) {
  if (factor <= 0.0) {
    return Number.NEGATIVE_INFINITY;
  }
  if (!Number.isFinite(logSurvival) && logSurvival < 0) {
    return Number.NEGATIVE_INFINITY;
  }
  if (factor >= 1.0 - EPSILON) {
    return logSurvival;
  }
  return logSurvival + Math.log(factor);
}

function committeePartialSurvivalFromDistribution(distribution, committeeSize, slotsInEpoch) {
  const partialSurvival = new Array(slotsInEpoch + 1).fill(0.0);
  partialSurvival[0] = 1.0;

  if (!distribution.length) {
    for (let slot = 1; slot <= slotsInEpoch; slot += 1) {
      partialSurvival[slot] = 1.0;
    }
    return partialSurvival;
  }

  const maxAllowed = maxCensorsAllowed(committeeSize);
  for (const [committeeCensors, probability] of distribution) {
    if (committeeCensors > maxAllowed) {
      for (let slot = 1; slot <= slotsInEpoch; slot += 1) {
        partialSurvival[slot] += probability;
      }
      continue;
    }

    const miss = committeeCensors / committeeSize;
    let missPower = 1.0;
    for (let slot = 1; slot <= slotsInEpoch; slot += 1) {
      missPower *= miss;
      partialSurvival[slot] += probability * missPower;
    }
  }

  return partialSurvival;
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

class SimulationModel {
  constructor(cfg) {
    this.cfg = cfg;
    const rawCensors = Math.round(cfg.base_sequencers * cfg.censor_fraction);
    this.initialCensors = Math.min(Math.max(rawCensors, 0), cfg.base_sequencers);
    this.initialHonest = cfg.base_sequencers - this.initialCensors;
    this.cache = new Map();
  }

  pEthereumCanonicalSlot(userSeq, maliciousSeq = 0) {
    const honestSeq = this.initialHonest + userSeq;
    const censorSeq = this.initialCensors + maliciousSeq;
    const totalSeq = honestSeq + censorSeq;
    if (totalSeq <= 0 || honestSeq <= censorSeq) {
      return 0.0;
    }
    return clampProbability(honestSeq / totalSeq);
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

  committeePartialSurvival(userSeq, maliciousSeq, slotsInEpoch) {
    const key = `partial|${userSeq}|${maliciousSeq}|${slotsInEpoch}`;
    if (this.cache.has(key)) {
      return this.cache.get(key);
    }

    const totalSeq = this.cfg.base_sequencers + userSeq + maliciousSeq;
    const censorSeq = this.initialCensors + maliciousSeq;
    let partialSurvival;

    if (this.cfg.committee_size > totalSeq) {
      partialSurvival = new Array(slotsInEpoch + 1).fill(1.0);
      partialSurvival[0] = 1.0;
    } else {
      partialSurvival = committeePartialSurvivalFromDistribution(
        this.committeeDistribution(totalSeq, censorSeq, this.cfg.committee_size),
        this.cfg.committee_size,
        slotsInEpoch
      );
    }

    this.cache.set(key, partialSurvival);
    return partialSurvival;
  }
}

function geometricPartialSurvivalFromSlotProbability(pSlot, slotsInEpoch) {
  const miss = 1.0 - pSlot;
  const partialSurvival = new Array(slotsInEpoch + 1).fill(1.0);
  let missPower = 1.0;
  for (let slot = 1; slot <= slotsInEpoch; slot += 1) {
    missPower *= miss;
    partialSurvival[slot] = missPower;
  }
  return partialSurvival;
}

function singleProposerPartialSurvivalCurve(
  model,
  inclusionModel,
  activeUser,
  activeMalicious,
  slotsInEpoch
) {
  const pSlot = inclusionModel.slotProbability(model, activeUser, activeMalicious);
  return geometricPartialSurvivalFromSlotProbability(pSlot, slotsInEpoch);
}

// Aztec committees sample from a validator set that is lagged by a fixed number of epochs.
function scheduledUserSequencersForEpoch(cfg, epochIdx, lagEpochs = 0) {
  const effectiveEpochIdx = Math.max(0, epochIdx - lagEpochs);
  return effectiveEpochIdx * cfg.max_new_sequencers_per_epoch;
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
  const fullCycles = Math.floor(elapsedSlots / cycleSlots);
  const residualSlots = elapsedSlots % cycleSlots;
  const openExtra =
    residualSlots <= 0 ? 0 : Math.max(0, openSlots + residualSlots - cycleSlots);
  const openBase = openSlots - openExtra;
  const nonOpenExtra = residualSlots - openExtra;
  const nonOpenBase = (cycleSlots - openSlots) - nonOpenExtra;

  const totalSurvival =
    (nonOpenBase * (miss ** fullCycles))
    + ((openBase + nonOpenExtra) * (miss ** (fullCycles + 1)))
    + (openExtra * (miss ** (fullCycles + 2)));

  return clampProbability(totalSurvival / cycleSlots);
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

function referenceTargetPointFromPartialSurvival(cfg, partialSurvival) {
  const slotsPerCycle = partialSurvival.length - 1;
  const pCycleCensor = partialSurvival[slotsPerCycle];
  if (pCycleCensor >= 1.0 - EPSILON) {
    return {
      pCycleCensor,
      targetSlots: Number.POSITIVE_INFINITY,
      targetDays: Number.POSITIVE_INFINITY,
    };
  }

  const targetSurvival = 1.0 - (cfg.target_inclusion_percent / 100.0);
  const logTargetSurvival = Math.log(targetSurvival);
  let fullCyclesBeforeSearch = 0;
  let logCycleSurvival = Number.NEGATIVE_INFINITY;

  if (pCycleCensor > 0.0) {
    logCycleSurvival = Math.log(pCycleCensor);
    fullCyclesBeforeSearch = Math.max(0, Math.ceil(logTargetSurvival / logCycleSurvival) - 1);
  }

  const logPrefixSurvival =
    fullCyclesBeforeSearch === 0 ? 0.0 : fullCyclesBeforeSearch * logCycleSurvival;

  for (let slot = 1; slot <= slotsPerCycle; slot += 1) {
    const partial = partialSurvival[slot];
    const logCandidateSurvival =
      partial <= 0.0 ? Number.NEGATIVE_INFINITY : logPrefixSurvival + Math.log(partial);

    if (logCandidateSurvival <= logTargetSurvival + 1e-12) {
      const targetSlots = (fullCyclesBeforeSearch * slotsPerCycle) + slot;
      return {
        pCycleCensor,
        targetSlots,
        targetDays: targetSlots * cfg.slot_seconds / (24 * 3600),
      };
    }
  }

  const targetSlots = (fullCyclesBeforeSearch + 1) * slotsPerCycle;
  return {
    pCycleCensor,
    targetSlots,
    targetDays: targetSlots * cfg.slot_seconds / (24 * 3600),
  };
}

function referenceCommitteeCurvePoint(cfg, model, censorCount) {
  const partialSurvival = committeePartialSurvivalFromDistribution(
    model.committeeDistribution(cfg.base_sequencers, censorCount, cfg.committee_size),
    cfg.committee_size,
    cfg.epoch_slots
  );
  const point = referenceTargetPointFromPartialSurvival(cfg, partialSurvival);
  return {
    ...point,
    inclusionProbability: 1.0 - point.pCycleCensor,
  };
}

function referenceEthereumCurvePoint(cfg, censorCount) {
  const totalSeq = cfg.base_sequencers;
  const honestSeq = Math.max(0, totalSeq - censorCount);
  const pSlot =
    totalSeq <= 0 || honestSeq <= censorCount ? 0.0 : clampProbability(honestSeq / totalSeq);
  const partialSurvival = geometricPartialSurvivalFromSlotProbability(pSlot, cfg.epoch_slots);
  const point = referenceTargetPointFromPartialSurvival(cfg, partialSurvival);
  return {
    ...point,
    inclusionProbability: pSlot,
  };
}

function referenceHoverText(totalSeq, censorCount, inclusionProbability, probabilityLabel, label = "") {
  const prefix = label ? `${label}<br>` : "";
  return `${prefix}Censoring fraction: ${((100 * censorCount) / totalSeq).toFixed(2)}%`
    + `<br>Censoring sequencers: ${censorCount.toLocaleString()} / ${totalSeq.toLocaleString()}`
    + `<br>${probabilityLabel}: ${inclusionProbability.toFixed(12)}`;
}

function currentReferencePoint(totalSeq, censorCount, point, probabilityLabel, label) {
  if (!Number.isFinite(point.targetDays)) {
    return null;
  }
  return {
    fraction: censorCount / totalSeq,
    delayDays: point.targetDays,
    hover: referenceHoverText(
      totalSeq,
      censorCount,
      point.inclusionProbability,
      probabilityLabel,
      label
    ),
  };
}

function buildReferenceCurves(cfg, model) {
  const fractions = [];
  const committeeDelayDays = [];
  const committeeHover = [];
  const ethereumDelayDays = [];
  const ethereumHover = [];

  for (let censorCount = 0; censorCount <= cfg.base_sequencers; censorCount += 1) {
    const committeePoint = referenceCommitteeCurvePoint(cfg, model, censorCount);
    const ethereumPoint = referenceEthereumCurvePoint(cfg, censorCount);
    fractions.push(censorCount / cfg.base_sequencers);
    committeeDelayDays.push(Number.isFinite(committeePoint.targetDays) ? committeePoint.targetDays : null);
    committeeHover.push(
      referenceHoverText(
        cfg.base_sequencers,
        censorCount,
        committeePoint.inclusionProbability,
        "Epoch inclusion probability"
      )
    );
    ethereumDelayDays.push(Number.isFinite(ethereumPoint.targetDays) ? ethereumPoint.targetDays : null);
    ethereumHover.push(
      referenceHoverText(
        cfg.base_sequencers,
        censorCount,
        ethereumPoint.inclusionProbability,
        "Slot inclusion probability"
      )
    );
  }

  const currentCensorCount = Math.min(
    cfg.base_sequencers,
    Math.max(0, Math.round(cfg.base_sequencers * cfg.censor_fraction))
  );
  const currentCommitteePoint = referenceCommitteeCurvePoint(cfg, model, currentCensorCount);
  const currentEthereumPoint = referenceEthereumCurvePoint(cfg, currentCensorCount);

  return {
    fractions,
    committee: {
      delayDays: committeeDelayDays,
      hover: committeeHover,
      current: currentReferencePoint(
        cfg.base_sequencers,
        currentCensorCount,
        currentCommitteePoint,
        "Epoch inclusion probability",
        "Current config (Committee)"
      ),
    },
    ethereum: {
      delayDays: ethereumDelayDays,
      hover: ethereumHover,
      current: currentReferencePoint(
        cfg.base_sequencers,
        currentCensorCount,
        currentEthereumPoint,
        "Slot inclusion probability",
        "Current config (Ethereum)"
      ),
    },
  };
}

function buildTimeSeries(cfg, model) {
  const maxSlots = maxHorizonSlots(cfg);
  const maxEpochs = Math.floor((maxSlots + cfg.epoch_slots - 1) / cfg.epoch_slots);
  const usdPerSeq = cfg.stake_per_sequencer_token * cfg.token_usd;

  const days = [0.0];
  const invested = [0.0];
  const userSeq = [0];
  const ethereumProbs = [0.0];
  const comProbs = [0.0];
  const comEhProbs = [0.0];
  const ethereumEffPerSlot = [0.0];
  const comEffPerSlot = [0.0];
  const comEhEffPerSlot = [0.0];

  let logSurvivalEthereum = 0.0;
  let logSurvivalCom = 0.0;
  let elapsedSlots = 0;
  let previousEhSurvival = 1.0;

  for (let epochIdx = 0; epochIdx < maxEpochs; epochIdx += 1) {
    const slotsInEpoch = Math.min(cfg.epoch_slots, maxSlots - elapsedSlots);
    if (slotsInEpoch <= 0) {
      break;
    }
    const activeUserEthereum = scheduledUserSequencersForEpoch(cfg, epochIdx);
    const activeUserCom = scheduledUserSequencersForEpoch(cfg, epochIdx, cfg.validator_set_lag_epochs);
    const activeMaliciousEthereum = maliciousSequencersForHonest(cfg, activeUserEthereum);
    const activeMaliciousCom = maliciousSequencersForHonest(cfg, activeUserCom);
    const partialEthereum = singleProposerPartialSurvivalCurve(
      model,
      ETHEREUM_INCLUSION_MODEL,
      activeUserEthereum,
      activeMaliciousEthereum,
      slotsInEpoch
    );
    const partialCom = model.committeePartialSurvival(activeUserCom, activeMaliciousCom, slotsInEpoch);
    const epochStartLogSurvivalEthereum = logSurvivalEthereum;
    const epochStartLogSurvivalCom = logSurvivalCom;
    const currentUserSeq = activeUserEthereum;
    const currentInvestedUsd = currentUserSeq * usdPerSeq;

    for (let slot = 1; slot <= slotsInEpoch; slot += 1) {
      elapsedSlots += 1;
      logSurvivalEthereum = logMultiplySurvival(
        epochStartLogSurvivalEthereum,
        partialEthereum[slot]
      );
      logSurvivalCom = logMultiplySurvival(epochStartLogSurvivalCom, partialCom[slot]);

      days.push((elapsedSlots * cfg.slot_seconds) / (24 * 3600));
      invested.push(currentInvestedUsd);
      userSeq.push(currentUserSeq);

      const ehSurvival = escapeHatchSurvivalProbability(cfg, elapsedSlots);
      const logSurvivalEh = ehSurvival <= 0.0 ? Number.NEGATIVE_INFINITY : Math.log(ehSurvival);
      const logSurvivalComEh =
        !Number.isFinite(logSurvivalCom) || !Number.isFinite(logSurvivalEh)
          ? Number.NEGATIVE_INFINITY
          : logSurvivalCom + logSurvivalEh;

      const pEthereum = horizonProbabilityFromLogSurvival(logSurvivalEthereum);
      const pCom = horizonProbabilityFromLogSurvival(logSurvivalCom);
      const pComEh = horizonProbabilityFromLogSurvival(logSurvivalComEh);
      ethereumProbs.push(pEthereum);
      comProbs.push(pCom);
      comEhProbs.push(pComEh);

      const missEthereum = partialEthereum[slot] / partialEthereum[slot - 1];
      const missCom = partialCom[slot] / partialCom[slot - 1];
      const missEh = previousEhSurvival <= 0.0 ? 1.0 : ehSurvival / previousEhSurvival;
      ethereumEffPerSlot.push(clampProbability(1.0 - missEthereum));
      comEffPerSlot.push(clampProbability(1.0 - missCom));
      comEhEffPerSlot.push(clampProbability(1.0 - (missCom * missEh)));
      previousEhSurvival = ehSurvival;
    }
  }

  return {
    days,
    invested,
    userSeq,
    cumulative: {
      ethereum: ethereumProbs,
      com: comProbs,
      comEh: comEhProbs,
    },
    perSlot: {
      ethereum: ethereumEffPerSlot,
      com: comEffPerSlot,
      comEh: comEhEffPerSlot,
    },
  };
}

function targetResult(days, userSeq, probabilities, threshold, stakePerSequencerToken, tokenUsd) {
  for (let idx = 0; idx < probabilities.length && idx < days.length && idx < userSeq.length; idx += 1) {
    if (probabilities[idx] < threshold) {
      continue;
    }
    const stakeToken = userSeq[idx] * stakePerSequencerToken;
    return {
      days: days[idx],
      stakeToken,
      stakeUsd: stakeToken * tokenUsd,
    };
  }
  return { days: null, stakeToken: null, stakeUsd: null };
}

function runSimulation(cfg) {
  const model = new SimulationModel(cfg);
  const timeSeries = buildTimeSeries(cfg, model);
  const reference = buildReferenceCurves(cfg, model);
  const targetProbability = cfg.target_inclusion_percent / 100;
  const committeeTarget = targetResult(
    timeSeries.days,
    timeSeries.userSeq,
    timeSeries.cumulative.com,
    targetProbability,
    cfg.stake_per_sequencer_token,
    cfg.token_usd
  );
  const committeeEhTarget = targetResult(
    timeSeries.days,
    timeSeries.userSeq,
    timeSeries.cumulative.comEh,
    targetProbability,
    cfg.stake_per_sequencer_token,
    cfg.token_usd
  );
  const ethereumTarget = targetResult(
    timeSeries.days,
    timeSeries.userSeq,
    timeSeries.cumulative.ethereum,
    targetProbability,
    cfg.stake_per_sequencer_token,
    cfg.token_usd
  );
  const ehBondUsd = EH_BOND_TOKENS * cfg.token_usd;
  const ehWithdrawalTaxUsd = EH_WITHDRAWAL_TAX_TOKENS * cfg.token_usd;

  return {
    meta: {
      targetInclusionPercent: cfg.target_inclusion_percent,
      slotSeconds: cfg.slot_seconds,
      epochSlots: cfg.epoch_slots,
      maxNewSequencersPerEpoch: cfg.max_new_sequencers_per_epoch,
      usdPerSeq: cfg.stake_per_sequencer_token * cfg.token_usd,
      targetCommitteeDays: committeeTarget.days,
      targetCommitteeEhDays: committeeEhTarget.days,
      targetEthereumDays: ethereumTarget.days,
      targetCommitteeStakeToken: committeeTarget.stakeToken,
      targetCommitteeEhStakeToken: committeeEhTarget.stakeToken,
      targetEthereumStakeToken: ethereumTarget.stakeToken,
      targetCommitteeStakeUsd: committeeTarget.stakeUsd,
      targetCommitteeEhStakeUsd: committeeEhTarget.stakeUsd,
      targetEthereumStakeUsd: ethereumTarget.stakeUsd,
      escapeHatchBondUsd: ehBondUsd,
      escapeHatchWithdrawalTaxUsd: ehWithdrawalTaxUsd,
    },
    series: {
      time: timeSeries,
      reference,
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

function buildTimeHover(days, invested, userSeq) {
  return days.map((day, idx) => {
    const usd = Math.round(invested[idx]).toLocaleString();
    const seq = userSeq[idx].toLocaleString();
    return `Day ${day.toFixed(3)}<br>Invested: $${usd}<br>User sequencers: ${seq}`;
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
  const xaxis = {
    title: xLabel,
    tickmode: "array",
    tickvals: xTicks.values,
    ticktext: xTicks.labels,
    gridcolor: "#eef2f7",
    zeroline: false,
  };
  const yaxis = {
    title: yLabel,
    gridcolor: "#eef2f7",
    zeroline: false,
  };
  if (xRange !== undefined) {
    xaxis.range = xRange;
  }
  if (yRange !== undefined) {
    yaxis.range = yRange;
  }
  return {
    margin: { l: 70, r: 30, t: 50, b: 60 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    legend: { orientation: "h", y: 1.12, x: 0 },
    xaxis,
    yaxis,
  };
}

function renderCharts(output) {
  if (!window.Plotly) {
    throw new Error("Plotly failed to load. Check internet/CDN access.");
  }

  const time = output.series.time;
  const reference = output.series.reference;

  const cumulativeDays = time.days;
  const timeRange = [cumulativeDays[0], cumulativeDays[cumulativeDays.length - 1]];
  const cumulativeTicks = buildTimeStakeTicks(
    timeRange[0],
    timeRange[1],
    output.meta.slotSeconds,
    output.meta.epochSlots,
    output.meta.maxNewSequencersPerEpoch,
    output.meta.usdPerSeq
  );
  const cumulativeHover = buildTimeHover(time.days, time.invested, time.userSeq);
  const cumulativeLayout = baseLayout(
    "Elapsed days | invested stake in USD [user sequencers]",
    "Cumulative inclusion probability",
    timeRange,
    [0, 1],
    cumulativeTicks
  );
  const cumulativeData = [
    {
      x: cumulativeDays,
      y: time.cumulative.comEh,
      mode: "lines",
      name: "Committee + EH",
      text: cumulativeHover,
      hovertemplate: "%{text}<br>Cumulative probability: %{y:.6f}<extra>Committee + EH</extra>",
      line: { color: COMMITTEE_EH_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: cumulativeDays,
      y: time.cumulative.com,
      mode: "lines",
      name: "Committee",
      text: cumulativeHover,
      hovertemplate: "%{text}<br>Cumulative probability: %{y:.6f}<extra>Committee</extra>",
      line: { color: COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: cumulativeDays,
      y: time.cumulative.ethereum,
      mode: "lines",
      name: ETHEREUM_INCLUSION_MODEL.label,
      text: cumulativeHover,
      hovertemplate: `%{text}<br>Cumulative probability: %{y:.6f}<extra>${ETHEREUM_INCLUSION_MODEL.label}</extra>`,
      line: { color: ETHEREUM_COLOR, width: 2 },
      connectgaps: false,
    },
  ];

  const perSlotLayout = baseLayout(
    "Elapsed days | invested stake in USD [user sequencers]",
    "Current-slot inclusion probability while still waiting",
    timeRange,
    [0, 1],
    buildTimeStakeTicks(
      timeRange[0],
      timeRange[1],
      output.meta.slotSeconds,
      output.meta.epochSlots,
      output.meta.maxNewSequencersPerEpoch,
      output.meta.usdPerSeq
    )
  );
  const perSlotHover = cumulativeHover;

  const perSlotData = [
    {
      x: time.days,
      y: time.perSlot.comEh,
      mode: "lines",
      name: "Committee + EH",
      text: perSlotHover,
      hovertemplate: "%{text}<br>Current-slot inclusion: %{y:.6f}<extra>Committee + EH</extra>",
      line: { color: COMMITTEE_EH_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: time.days,
      y: time.perSlot.com,
      mode: "lines",
      name: "Committee",
      text: perSlotHover,
      hovertemplate: "%{text}<br>Current-slot inclusion: %{y:.6f}<extra>Committee</extra>",
      line: { color: COMMITTEE_COLOR, width: 2 },
      connectgaps: false,
    },
    {
      x: time.days,
      y: time.perSlot.ethereum,
      mode: "lines",
      name: ETHEREUM_INCLUSION_MODEL.label,
      text: perSlotHover,
      hovertemplate: `%{text}<br>Current-slot inclusion: %{y:.6f}<extra>${ETHEREUM_INCLUSION_MODEL.label}</extra>`,
      line: { color: ETHEREUM_COLOR, width: 2 },
      connectgaps: false,
    },
  ];

  const referenceXTicks = {
    values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    labels: ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
  };
  const referenceLayout = baseLayout(
    "Censorship fraction among the fixed base sequencer set",
    `Time to ${targetLabel(output.meta.targetInclusionPercent)} inclusion (days, log scale)`,
    [0, 1],
    undefined,
    referenceXTicks
  );
  referenceLayout.yaxis.type = "log";
  const referenceData = [
    {
      x: reference.fractions,
      y: reference.committee.delayDays,
      mode: "lines",
      name: "Committee static baseline",
      text: reference.committee.hover,
      hovertemplate: "%{text}<br>Target delay: %{y:.6f}d<extra>Committee static baseline</extra>",
      line: { color: COMMITTEE_COLOR, width: 2.5 },
      connectgaps: false,
    },
    {
      x: reference.fractions,
      y: reference.ethereum.delayDays,
      mode: "lines",
      name: "Ethereum static baseline",
      text: reference.ethereum.hover,
      hovertemplate: "%{text}<br>Target delay: %{y:.6f}d<extra>Ethereum static baseline</extra>",
      line: { color: ETHEREUM_COLOR, width: 2.5 },
      connectgaps: false,
    },
  ];

  if (reference.committee.current) {
    referenceData.push({
      x: [reference.committee.current.fraction],
      y: [reference.committee.current.delayDays],
      mode: "markers",
      name: "Current config (Committee)",
      text: [reference.committee.current.hover],
      hovertemplate: "%{text}<br>Target delay: %{y:.6f}d<extra>Current config (Committee)</extra>",
      marker: {
        color: REFERENCE_MARKER_COLOR,
        size: 9,
        symbol: "circle",
        line: { color: "#ffffff", width: 1.5 },
      },
    });
  }

  if (reference.ethereum.current) {
    referenceData.push({
      x: [reference.ethereum.current.fraction],
      y: [reference.ethereum.current.delayDays],
      mode: "markers",
      name: "Current config (Ethereum)",
      text: [reference.ethereum.current.hover],
      hovertemplate: "%{text}<br>Target delay: %{y:.6f}d<extra>Current config (Ethereum)</extra>",
      marker: {
        color: REFERENCE_MARKER_COLOR,
        size: 9,
        symbol: "diamond",
        line: { color: "#ffffff", width: 1.5 },
      },
    });
  }

  Plotly.react(cumulativeChartEl, cumulativeData, cumulativeLayout, plotConfig());
  Plotly.react(perSlotChartEl, perSlotData, perSlotLayout, plotConfig());
  Plotly.react(referenceBaselineChartEl, referenceData, referenceLayout, plotConfig());
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
    renderTargetCards(output.meta, cfg.max_horizon_days, targetLabel(cfg.target_inclusion_percent));
    setStatus("Simulation complete.");
  } catch (error) {
    clearTargetCards(targetLabel(cfg.target_inclusion_percent));
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
