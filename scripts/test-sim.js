#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

function loadApp() {
  const source = fs.readFileSync("web/app.js", "utf8");
  const inputs = new Map();
  const formEl = {
    elements: {
      namedItem(name) {
        if (!inputs.has(name)) {
          inputs.set(name, { value: "" });
        }
        return inputs.get(name);
      },
    },
  };

  const makeElement = () => ({
    textContent: "",
    style: {},
    addEventListener() {},
  });

  const elements = {
    "config-form": formEl,
    status: makeElement(),
    summary: makeElement(),
    "run-btn": makeElement(),
    "reset-btn": makeElement(),
    "csv-btn": makeElement(),
    "t90-committee-value": makeElement(),
    "t90-non-value": makeElement(),
    "t90-committee-cost": makeElement(),
    "t90-non-cost": makeElement(),
    "chart-cumulative": {},
    "chart-per-slot": {},
    "chart-delay": {},
  };

  const context = {
    Math,
    Number,
    String,
    Object,
    Array,
    Map,
    Set,
    JSON,
    URLSearchParams,
    Blob: class {},
    URL: {
      createObjectURL() {
        return "";
      },
      revokeObjectURL() {},
    },
    document: {
      getElementById(id) {
        return elements[id];
      },
      body: {
        appendChild() {},
      },
      createElement() {
        return {
          click() {},
          remove() {},
        };
      },
    },
    FormData: class {
      constructor(form) {
        this.form = form;
      }

      get(name) {
        const input = this.form.elements.namedItem(name);
        return input ? input.value : null;
      }
    },
    window: {
      location: {
        search: "",
        pathname: "/",
        hash: "",
      },
      history: {
        replaceState() {},
      },
      Plotly: null,
    },
    Plotly: {
      react() {},
    },
  };

  context.window.Plotly = context.Plotly;
  vm.createContext(context);
  vm.runInContext(source, context);
  return context;
}

function seedRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function firstIndexAtOrAbove(values, threshold) {
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] >= threshold) {
      return i;
    }
  }
  return null;
}

function assertUnitInterval(values, label) {
  for (const value of values) {
    assert(Number.isFinite(value), `${label} contains non-finite value`);
    assert(value >= -1e-12, `${label} contains value below 0: ${value}`);
    assert(value <= 1.0 + 1e-12, `${label} contains value above 1: ${value}`);
  }
}

function assertMonotonic(values, label) {
  for (let i = 1; i < values.length; i += 1) {
    assert(values[i] + 1e-12 >= values[i - 1], `${label} is not monotonic at index ${i}`);
  }
}

const tests = [];

function test(name, fn) {
  tests.push({ name, fn });
}

const app = loadApp();
const defaultCfg = app.getConfigFromForm();

test("rejects committee sizes that exceed base sequencers", () => {
  const cfg = {
    ...defaultCfg,
    base_sequencers: 100,
    committee_size: 101,
  };
  const errors = app.validateConfig(cfg);
  assert(
    errors.some((msg) => msg.includes("committee_size must be <= base_sequencers at simulation start")),
    "expected committee-size feasibility validation error"
  );
});

test("keeps committee probabilities stable for large committee sizes", () => {
  for (const committeeSize of [3200, 3500, 3727]) {
    const cfg = {
      ...defaultCfg,
      base_sequencers: 4000,
      censor_fraction: 0.1,
      committee_size: committeeSize,
      max_horizon_days: 0.1,
      honest_add_success_rate: 1.0,
    };
    const errors = app.validateConfig(cfg);
    assert.equal(errors.length, 0, `unexpected validation failure for committee_size=${committeeSize}`);
    const row0 = app.runSimulation(cfg).rows[0];
    assert(
      row0.p_committee_allows_honest > 0.999999,
      `committee gate should almost always allow with committee_size=${committeeSize}`
    );
    assert(
      row0["p_eff_0.1d_committee_per_slot"] > 0.85,
      `effective committee per-slot probability collapsed for committee_size=${committeeSize}`
    );
  }
});

test("aligns cumulative sequencer counts with the simulated epochs", () => {
  const out = app.runSimulation(defaultCfg);
  assert.equal(out.series.cumulativeUserSeq[1], 0, "first epoch should still run with 0 user sequencers");
  assert.equal(
    out.series.cumulativeUserSeq[2],
    defaultCfg.max_new_sequencers_per_epoch,
    "second epoch should use first onboarding increment"
  );

  const t90Index = firstIndexAtOrAbove(out.series.cumulativeNonProbs, 0.9);
  assert.notEqual(t90Index, null, "default config should reach non-committee T90");
  assert.equal(
    out.series.cumulativeUserSeq[t90Index],
    0,
    "T90 non-committee should map to the same sequencer count used for that probability point"
  );
  assert.equal(out.meta.t90NonCommitteeStakeToken, 0, "T90 non-committee stake should match sequencer count");
});

test("preserves probability invariants across random configs", () => {
  const rand = seedRandom(123456);
  const randomInt = (min, max) => Math.floor(min + rand() * (max - min + 1));
  const randomFloat = (min, max) => min + rand() * (max - min);

  for (let i = 0; i < 40; i += 1) {
    const baseSequencers = randomInt(100, 1500);
    const cfg = {
      ...defaultCfg,
      base_sequencers: baseSequencers,
      committee_size: randomInt(1, Math.min(200, baseSequencers)),
      censor_fraction: randomFloat(0, 1),
      slot_seconds: randomInt(12, 120),
      max_horizon_days: randomFloat(0.5, 15),
      epoch_slots: randomInt(4, 64),
      max_new_sequencers_per_epoch: randomInt(1, 16),
      honest_add_success_rate: randomFloat(0.1, 1.0),
      probability_near_one_margin: randomFloat(1e-5, 0.2),
    };

    const errors = app.validateConfig(cfg);
    assert.equal(errors.length, 0, `unexpected validation errors: ${errors.join(" | ")}`);
    const out = app.runSimulation(cfg);

    assertUnitInterval(out.series.cumulativeNonProbs, "cumulative non-committee");
    assertUnitInterval(out.series.cumulativeComProbs, "cumulative committee");
    assertUnitInterval(out.series.perSlotNonEffPerSlot, "per-slot non-committee");
    assertUnitInterval(out.series.perSlotComEffPerSlot, "per-slot committee");

    assertMonotonic(out.series.cumulativeNonProbs, "cumulative non-committee");
    assertMonotonic(out.series.cumulativeComProbs, "cumulative committee");
  }
});

let failures = 0;
for (const { name, fn } of tests) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`not ok - ${name}`);
    console.error(error.stack || String(error));
  }
}

if (failures > 0) {
  console.error(`${failures} test(s) failed`);
  process.exit(1);
}

console.log(`${tests.length} test(s) passed`);
