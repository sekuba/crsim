#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    base_sequencers: int = 4000
    stake_per_sequencer_token: int = 20_000
    token_usd: float = 0.02
    censor_fraction: float = 1

    committee_size: int = 24
    slot_seconds: int = 72

    horizon_days: int = 30
    epoch_slots: int = 32
    max_new_sequencers_per_epoch: int = 4

    delay_focus_x_min_usd: float = 1_000.0
    delay_focus_x_max_usd: float = 120_000.0
    delay_focus_y_min_hours: float = 0.03
    delay_focus_y_max_hours: float = 168 # 7d
    delay_focus_y_padding_ratio: float = 0.10
    # Probability chart x-zoom: hide tail where both curves are already ~1.
    probability_near_one_margin: float = 0.001

    @property
    def usd_per_user_seq(self) -> float:
        return self.stake_per_sequencer_token * self.token_usd

    @property
    def horizon_slots(self) -> int:
        return int((self.horizon_days * 24 * 3600) // self.slot_seconds)

    @property
    def max_user_sequencers(self) -> int:
        # Max active user sequencers reachable within the configured horizon.
        return (self.horizon_slots // self.epoch_slots) * self.max_new_sequencers_per_epoch

    @property
    def user_seq_values(self) -> list[int]:
        # Always start at 0 user sequencers; choose an auto step for readability/perf.
        max_seq = self.max_user_sequencers
        step = max(1, max_seq // 250)
        values = list(range(0, max_seq + 1, step))
        if values[-1] != max_seq:
            values.append(max_seq)
        return values


def max_censors_allowed(committee_size: int) -> int:
    # Honest proposer can include if censoring committee members are strictly below 1/3.
    return math.ceil(committee_size / 3) - 1


def hypergeom_cdf_max_successes(population: int, success_population: int, draws: int, max_successes: int) -> float:
    if draws > population:
        return 0.0
    denom = math.comb(population, draws)
    lo = max(0, draws - (population - success_population))
    hi = min(draws, success_population, max_successes)
    total = 0
    for x in range(lo, hi + 1):
        total += math.comb(success_population, x) * math.comb(population - success_population, draws - x)
    return total / denom


def committee_censor_distribution(
    cfg: Config, user_seq: int, committee_size: int | None = None
) -> list[tuple[int, float]]:
    k = cfg.committee_size if committee_size is None else committee_size
    total_seq = cfg.base_sequencers + user_seq
    censors = int(round(cfg.base_sequencers * cfg.censor_fraction))
    if k > total_seq:
        return []
    denom = math.comb(total_seq, k)
    lo = max(0, k - (total_seq - censors))
    hi = min(k, censors)
    dist: list[tuple[int, float]] = []
    for c in range(lo, hi + 1):
        p = (math.comb(censors, c) * math.comb(total_seq - censors, k - c)) / denom
        dist.append((c, p))
    return dist


def p_non_committee_slot(cfg: Config, user_seq: int) -> float:
    total_seq = cfg.base_sequencers + user_seq
    censors = int(round(cfg.base_sequencers * cfg.censor_fraction))
    honest_non_user = cfg.base_sequencers - censors
    return (user_seq + honest_non_user) / total_seq


def geometric_sum(r: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if abs(1.0 - r) < 1e-15:
        return float(n)
    return (1.0 - (r ** n)) / (1.0 - r)


def epoch_survival_factor_and_tail_sum(m: int, p_non: float, p_allow: float, committee_mode: bool) -> tuple[float, float]:
    # f: survival multiplier after m slots in this epoch.
    # g: sum_{j=0..m-1} P(T > start+j) / P(T > start).
    miss = 1.0 - p_non
    if committee_mode:
        f = (1.0 - p_allow) + p_allow * (miss ** m)
        g = m * (1.0 - p_allow) + p_allow * geometric_sum(miss, m)
    else:
        f = miss ** m
        g = geometric_sum(miss, m)
    return f, g


def committee_epoch_factors(cfg: Config, user_seq: int, m: int, committee_size: int | None = None) -> tuple[float, float, float]:
    k = cfg.committee_size if committee_size is None else committee_size
    dist = committee_censor_distribution(cfg, user_seq, committee_size=k)
    if not dist:
        return 1.0, float(m), 0.0

    max_c = max_censors_allowed(k)
    f_total = 0.0
    g_total = 0.0
    p_allow = 0.0
    for c, p in dist:
        allow = c <= max_c
        if allow:
            p_allow += p
            h = k - c
            p_non_from_committee = h / k
            miss = 1.0 - p_non_from_committee
            f = miss ** m
            g = geometric_sum(miss, m)
        else:
            f = 1.0
            g = float(m)
        f_total += p * f
        g_total += p * g
    return f_total, g_total, p_allow


def horizon_prob_with_churn(
    cfg: Config,
    target_user_seq: int,
    committee_mode: bool,
    slots: int,
    committee_size: int | None = None,
) -> float:
    log_survival = 0.0
    epochs = (slots + cfg.epoch_slots - 1) // cfg.epoch_slots
    for e in range(epochs):
        m = min(cfg.epoch_slots, slots - e * cfg.epoch_slots)
        active_user = min(target_user_seq, e * cfg.max_new_sequencers_per_epoch)
        if committee_mode:
            f, _, _ = committee_epoch_factors(cfg, active_user, m, committee_size=committee_size)
        else:
            p_non = p_non_committee_slot(cfg, active_user)
            miss = 1.0 - p_non
            f = miss ** m
        if f <= 0.0:
            return 1.0
        log_survival += math.log(f)
    p = -math.expm1(log_survival)
    return 0.0 if abs(p) < 1e-15 else p


def expected_delay_hours_with_churn(cfg: Config, target_user_seq: int, committee_mode: bool) -> float:
    # E[T] for time-varying Bernoulli process:
    # E[T_slots] = sum_{s>=0} P(T_slots > s)
    epochs_to_full = math.ceil(target_user_seq / cfg.max_new_sequencers_per_epoch)
    full_onboard_slot = epochs_to_full * cfg.epoch_slots

    survival = 1.0
    expected_slots = 0.0

    # Onboarding phase (parameters change per epoch).
    for e in range(epochs_to_full):
        active_user = min(target_user_seq, e * cfg.max_new_sequencers_per_epoch)
        if committee_mode:
            f, g, _ = committee_epoch_factors(cfg, active_user, cfg.epoch_slots)
        else:
            p_non = p_non_committee_slot(cfg, active_user)
            miss = 1.0 - p_non
            f = miss ** cfg.epoch_slots
            g = geometric_sum(miss, cfg.epoch_slots)
        expected_slots += survival * g
        survival *= f

    # Steady state after full onboarding (same parameters every epoch).
    if committee_mode:
        f_final, g_final, _ = committee_epoch_factors(cfg, target_user_seq, cfg.epoch_slots)
    else:
        p_non_final = p_non_committee_slot(cfg, target_user_seq)
        miss_final = 1.0 - p_non_final
        f_final = miss_final ** cfg.epoch_slots
        g_final = geometric_sum(miss_final, cfg.epoch_slots)
    if f_final >= 1.0 - 1e-15:
        return float("inf")
    expected_slots += survival * g_final / (1.0 - f_final)
    return expected_slots * (cfg.slot_seconds / 3600.0)


def effective_per_slot_with_churn(
    cfg: Config,
    target_user_seq: int,
    committee_mode: bool,
    slots: int,
    committee_size: int | None = None,
) -> float:
    if slots <= 0:
        return 0.0
    log_survival = 0.0
    epochs = (slots + cfg.epoch_slots - 1) // cfg.epoch_slots
    for e in range(epochs):
        m = min(cfg.epoch_slots, slots - e * cfg.epoch_slots)
        active_user = min(target_user_seq, e * cfg.max_new_sequencers_per_epoch)
        if committee_mode:
            f, _, _ = committee_epoch_factors(cfg, active_user, m, committee_size=committee_size)
        else:
            p_non = p_non_committee_slot(cfg, active_user)
            miss = 1.0 - p_non
            f = miss ** m
        if f <= 0.0:
            return 1.0
        log_survival += math.log(f)
    # Equivalent constant per-slot probability over the same horizon.
    p = -math.expm1(log_survival / slots)
    return 0.0 if abs(p) < 1e-15 else p


def _polyline(xs: list[float], ys: list[float], x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    pts: list[str] = []
    for x, y in zip(xs, ys):
        px = 80 + (x - x_min) / (x_max - x_min) * (1100 - 80) if x_max > x_min else 80
        py = 560 - (y - y_min) / (y_max - y_min) * (560 - 40) if y_max > y_min else 560
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def write_svg_line_chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    xs: list[float],
    series: list[tuple[str, list[float], str, str | None]],
    y_min: float | None = None,
    y_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_tick_decimals: int = 2,
    x_ticks_usd_and_seq: bool = False,
    usd_per_seq: float = 1.0,
) -> None:
    x_min = min(xs) if x_min is None else x_min
    x_max = max(xs) if x_max is None else x_max

    filtered: list[tuple[list[float], list[float]]] = []
    all_y: list[float] = []
    for _, ys, _, _ in series:
        pairs = [(x, y) for x, y in zip(xs, ys) if x_min <= x <= x_max]
        fxs = [p[0] for p in pairs]
        fys = [p[1] for p in pairs]
        filtered.append((fxs, fys))
        all_y.extend(fys)

    if not all_y:
        raise ValueError("No points in selected x-range")

    y_min = min(all_y) if y_min is None else y_min
    y_max = max(all_y) if y_max is None else y_max

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="640">',
        '<rect x="0" y="0" width="1200" height="640" fill="white"/>',
        '<line x1="80" y1="560" x2="1100" y2="560" stroke="black" stroke-width="2"/>',
        '<line x1="80" y1="560" x2="80" y2="40" stroke="black" stroke-width="2"/>',
        f'<text x="600" y="24" font-family="sans-serif" font-size="20" text-anchor="middle">{title}</text>',
        f'<text x="600" y="620" font-family="sans-serif" font-size="16" text-anchor="middle">{x_label}</text>',
        f'<text x="20" y="300" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90,20,300)">{y_label}</text>',
    ]

    for i, (name, _, color, dash) in enumerate(series):
        fxs, fys = filtered[i]
        if not fxs:
            continue
        capped = [min(max(y, y_min), y_max) for y in fys]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} '
            f'points="{_polyline(fxs, capped, x_min, x_max, y_min, y_max)}"/>'
        )
        lines.append(f'<text x="860" y="{70 + i * 24}" font-family="sans-serif" font-size="14" fill="{color}">{name}</text>')

    for i in range(6):
        tx_val = x_min + (x_max - x_min) * i / 5
        tx = 80 + (1100 - 80) * i / 5
        if x_ticks_usd_and_seq:
            seq = int(round(tx_val / usd_per_seq))
            tick = f"${int(round(tx_val)):,} [{seq}]"
        else:
            tick = f"{int(round(tx_val)):,}"
        lines.append(f'<line x1="{tx:.2f}" y1="560" x2="{tx:.2f}" y2="566" stroke="black"/>')
        lines.append(f'<text x="{tx:.2f}" y="586" font-family="sans-serif" font-size="12" text-anchor="middle">{tick}</text>')

    for i in range(6):
        ty = 560 - (560 - 40) * i / 5
        val = y_min + (y_max - y_min) * i / 5
        lines.append(f'<line x1="74" y1="{ty:.2f}" x2="80" y2="{ty:.2f}" stroke="black"/>')
        lines.append(f'<text x="68" y="{ty + 4:.2f}" font-family="sans-serif" font-size="12" text-anchor="end">{val:.{y_tick_decimals}f}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines))


def run(cfg: Config) -> None:
    slots = cfg.horizon_slots
    usd_per_seq = cfg.usd_per_user_seq
    horizon_label = f"{cfg.horizon_days} day" if cfg.horizon_days == 1 else f"{cfg.horizon_days} days"
    horizon_tag = f"{cfg.horizon_days}d"
    p_horizon_non_key = f"p_{horizon_tag}_non_committee_with_churn_exact"
    p_horizon_com_key = f"p_{horizon_tag}_committee_with_churn_exact"
    p_eff_non_key = f"p_eff_{horizon_tag}_non_committee_per_slot"
    p_eff_com_key = f"p_eff_{horizon_tag}_committee_per_slot"

    rows: list[dict[str, float]] = []
    for user_seq in cfg.user_seq_values:
        invested_token = user_seq * cfg.stake_per_sequencer_token
        invested_usd = invested_token * cfg.token_usd
        _, _, p_allow = committee_epoch_factors(cfg, user_seq, cfg.epoch_slots)
        p_horizon_non = horizon_prob_with_churn(cfg, user_seq, committee_mode=False, slots=slots)
        p_horizon_com = horizon_prob_with_churn(cfg, user_seq, committee_mode=True, slots=slots)

        rows.append(
            {
                "user_sequencers": user_seq,
                "invested_stake_token": invested_token,
                "invested_stake_usd": invested_usd,
                "p_committee_allows_honest": p_allow,
                p_horizon_non_key: p_horizon_non,
                p_horizon_com_key: p_horizon_com,
                p_eff_non_key: effective_per_slot_with_churn(cfg, user_seq, committee_mode=False, slots=slots),
                p_eff_com_key: effective_per_slot_with_churn(cfg, user_seq, committee_mode=True, slots=slots),
                "expected_hours_non_committee": expected_delay_hours_with_churn(cfg, user_seq, committee_mode=False),
                "expected_hours_committee": expected_delay_hours_with_churn(cfg, user_seq, committee_mode=True),
            }
        )

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    main_csv = results_dir / "cr_simulation.csv"
    with main_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    invested = [float(r["invested_stake_usd"]) for r in rows]
    non_probs = [r[p_horizon_non_key] for r in rows]
    com_probs = [r[p_horizon_com_key] for r in rows]

    near_one_threshold = 1.0 - cfg.probability_near_one_margin
    unsaturated_indices = [
        i for i, (p_non, p_com) in enumerate(zip(non_probs, com_probs))
        if (p_non < near_one_threshold or p_com < near_one_threshold)
    ]
    if unsaturated_indices:
        x_zoom_idx = min(len(invested) - 1, max(unsaturated_indices) + 1)
        probability_x_max = invested[x_zoom_idx]
    else:
        probability_x_max = invested[-1]

    write_svg_line_chart(
        path=figures_dir / f"cr_{horizon_tag}_probability_vs_stake.svg",
        title=f"CR vs Invested Stake Over {horizon_label.title()}",
        x_label="Invested stake in USD [user sequencers]",
        y_label=f"Inclusion probability within {horizon_label}",
        xs=invested,
        series=[
            ("Committee", com_probs, "#B02E0C", "6,4"),
            ("Non-committee", non_probs, "#0B6E4F", None),
        ],
        x_max=probability_x_max,
        y_min=0.0,
        y_max=1.0,
        y_tick_decimals=2,
        x_ticks_usd_and_seq=True,
        usd_per_seq=usd_per_seq,
    )

    write_svg_line_chart(
        path=figures_dir / "cr_per_slot_probability_vs_stake.svg",
        title=f"Effective Per-slot Inclusion vs Invested Stake (derived from {horizon_tag} horizon)",
        x_label="Invested stake in USD [user sequencers]",
        y_label="Effective per-slot inclusion probability",
        xs=invested,
        series=[
            ("Committee", [r[p_eff_com_key] for r in rows], "#B02E0C", "6,4"),
            ("Non-committee", [r[p_eff_non_key] for r in rows], "#0B6E4F", None),
        ],
        y_min=0.0,
        y_max=1.0,
        y_tick_decimals=2,
        x_ticks_usd_and_seq=True,
        usd_per_seq=usd_per_seq,
    )

    delay_rows = [r for r in rows if cfg.delay_focus_x_min_usd <= r["invested_stake_usd"] <= cfg.delay_focus_x_max_usd]
    if not delay_rows:
        delay_rows = rows
    finite_delay_values = [
        v
        for r in delay_rows
        for v in (r["expected_hours_non_committee"], r["expected_hours_committee"])
        if math.isfinite(v)
    ]
    if finite_delay_values:
        delay_y_max = max(finite_delay_values) * (1.0 + cfg.delay_focus_y_padding_ratio)
    else:
        delay_y_max = cfg.delay_focus_y_min_hours * 2.0
    delay_y_max = min(delay_y_max, cfg.delay_focus_y_max_hours)
    if delay_y_max <= cfg.delay_focus_y_min_hours:
        delay_y_max = cfg.delay_focus_y_min_hours * 2.0

    write_svg_line_chart(
        path=figures_dir / "cr_expected_delay_vs_stake.svg",
        title="Expected Inclusion Delay vs Invested Stake",
        x_label="Invested stake in USD [user sequencers]",
        y_label="Expected time to inclusion (hours)",
        xs=invested,
        series=[
            ("Committee", [r["expected_hours_committee"] for r in rows], "#B02E0C", "6,4"),
            ("Non-committee", [r["expected_hours_non_committee"] for r in rows], "#0B6E4F", None),
        ],
        x_min=cfg.delay_focus_x_min_usd,
        x_max=cfg.delay_focus_x_max_usd,
        y_min=cfg.delay_focus_y_min_hours,
        y_max=delay_y_max,
        y_tick_decimals=2,
        x_ticks_usd_and_seq=True,
        usd_per_seq=usd_per_seq,
    )

    print(f"Wrote {main_csv}")
    print("Wrote figures:")
    print(f" - figures/cr_{horizon_tag}_probability_vs_stake.svg")
    print(" - figures/cr_per_slot_probability_vs_stake.svg")
    print(" - figures/cr_expected_delay_vs_stake.svg")
    print(f"Slots in horizon: {slots}")


if __name__ == "__main__":
    run(Config())
