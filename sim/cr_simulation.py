#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

EPSILON = 1e-15


@dataclass(frozen=True)
class Config:
    base_sequencers: int = 4000
    stake_per_sequencer_token: int = 20_000
    token_usd: float = 0.02
    censor_fraction: float = 1

    committee_size: int = 24
    slot_seconds: int = 72

    horizon_days: int = 100
    epoch_slots: int = 32
    max_new_sequencers_per_epoch: int = 4

    # Probability chart x-zoom: hide tail where both curves are already ~1.
    probability_near_one_margin: float = 0.001

    def validate(self) -> None:
        if self.base_sequencers < 0:
            raise ValueError("base_sequencers must be >= 0")
        if self.stake_per_sequencer_token <= 0:
            raise ValueError("stake_per_sequencer_token must be > 0")
        if self.token_usd <= 0:
            raise ValueError("token_usd must be > 0")
        if not (0.0 <= self.censor_fraction <= 1.0):
            raise ValueError("censor_fraction must be in [0, 1]")
        if self.committee_size <= 0:
            raise ValueError("committee_size must be > 0")
        if self.slot_seconds <= 0:
            raise ValueError("slot_seconds must be > 0")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")
        if self.epoch_slots <= 0:
            raise ValueError("epoch_slots must be > 0")
        if self.max_new_sequencers_per_epoch <= 0:
            raise ValueError("max_new_sequencers_per_epoch must be > 0")
        if not (0.0 < self.probability_near_one_margin < 1.0):
            raise ValueError("probability_near_one_margin must be in (0, 1)")

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
        if max_seq <= 0:
            return [0]
        step = max(1, max_seq // 250)
        values = list(range(0, max_seq + 1, step))
        if values[-1] != max_seq:
            values.append(max_seq)
        return values


def max_censors_allowed(committee_size: int) -> int:
    # Honest proposer can include if censoring committee members are strictly below 1/3.
    return (committee_size - 1) // 3


def clamp_probability(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


class SimulationModel:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        censors = int(round(cfg.base_sequencers * cfg.censor_fraction))
        self.initial_censors = min(max(censors, 0), cfg.base_sequencers)
        self.initial_honest = cfg.base_sequencers - self.initial_censors
        self._committee_factor_cache: dict[tuple[int, int, int], tuple[float, float, float]] = {}

    def p_non_committee_slot(self, user_seq: int) -> float:
        total_seq = self.cfg.base_sequencers + user_seq
        if total_seq <= 0:
            return 0.0
        return clamp_probability((user_seq + self.initial_honest) / total_seq)

    def _committee_distribution(self, total_seq: int, committee_size: int) -> list[tuple[int, float]]:
        if committee_size > total_seq:
            return []

        censor_seq = self.initial_censors
        honest_seq = total_seq - censor_seq
        lo = max(0, committee_size - honest_seq)
        hi = min(committee_size, censor_seq)
        if lo > hi:
            return []

        log_p_lo = (
            _log_choose(censor_seq, lo)
            + _log_choose(honest_seq, committee_size - lo)
            - _log_choose(total_seq, committee_size)
        )
        p = math.exp(log_p_lo)
        dist: list[tuple[int, float]] = []
        for c in range(lo, hi + 1):
            dist.append((c, p))
            if c == hi:
                break
            numer = (censor_seq - c) * (committee_size - c)
            denom = (c + 1) * (honest_seq - committee_size + c + 1)
            p *= numer / denom
        return dist

    def committee_epoch_factors(
        self,
        user_seq: int,
        slots_in_epoch: int,
        committee_size: int | None = None,
    ) -> tuple[float, float, float]:
        k = self.cfg.committee_size if committee_size is None else committee_size
        cache_key = (user_seq, slots_in_epoch, k)
        cached = self._committee_factor_cache.get(cache_key)
        if cached is not None:
            return cached

        total_seq = self.cfg.base_sequencers + user_seq
        if k > total_seq:
            result = (1.0, float(slots_in_epoch), 0.0)
            self._committee_factor_cache[cache_key] = result
            return result

        max_c = max_censors_allowed(k)
        p_allow = 0.0
        f_total = 0.0
        g_total = 0.0
        mass = 0.0

        for censor_committee, p in self._committee_distribution(total_seq, k):
            mass += p
            miss = censor_committee / k
            if censor_committee <= max_c:
                p_allow += p
                f = miss ** slots_in_epoch
                g = geometric_sum(miss, slots_in_epoch)
            else:
                f = 1.0
                g = float(slots_in_epoch)
            f_total += p * f
            g_total += p * g

        if mass <= EPSILON:
            result = (1.0, float(slots_in_epoch), 0.0)
        else:
            # Normalize minor floating-point drift from recurrence accumulation.
            inv_mass = 1.0 / mass
            result = (
                f_total * inv_mass,
                g_total * inv_mass,
                clamp_probability(p_allow * inv_mass),
            )
        self._committee_factor_cache[cache_key] = result
        return result


def geometric_sum(r: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if abs(1.0 - r) < EPSILON:
        return float(n)
    return (1.0 - (r**n)) / (1.0 - r)


def _epoch_factors(
    model: SimulationModel,
    active_user: int,
    slots_in_epoch: int,
    committee_mode: bool,
    committee_size: int | None = None,
) -> tuple[float, float]:
    if committee_mode:
        f, g, _ = model.committee_epoch_factors(
            user_seq=active_user,
            slots_in_epoch=slots_in_epoch,
            committee_size=committee_size,
        )
        return f, g

    p_non = model.p_non_committee_slot(active_user)
    miss = 1.0 - p_non
    return miss**slots_in_epoch, geometric_sum(miss, slots_in_epoch)


def _active_user_for_epoch(cfg: Config, target_user_seq: int, epoch_idx: int) -> int:
    return min(target_user_seq, epoch_idx * cfg.max_new_sequencers_per_epoch)


def _horizon_log_survival(
    cfg: Config,
    model: SimulationModel,
    target_user_seq: int,
    committee_mode: bool,
    slots: int,
    committee_size: int | None = None,
) -> float:
    if slots <= 0:
        return 0.0

    log_survival = 0.0
    epochs = (slots + cfg.epoch_slots - 1) // cfg.epoch_slots
    for epoch_idx in range(epochs):
        start_slot = epoch_idx * cfg.epoch_slots
        slots_in_epoch = min(cfg.epoch_slots, slots - start_slot)
        active_user = _active_user_for_epoch(cfg, target_user_seq, epoch_idx)
        f, _ = _epoch_factors(
            model=model,
            active_user=active_user,
            slots_in_epoch=slots_in_epoch,
            committee_mode=committee_mode,
            committee_size=committee_size,
        )
        if f <= 0.0:
            return float("-inf")
        log_survival += math.log(f)
    return log_survival


def _horizon_probability_from_log_survival(log_survival: float) -> float:
    if math.isinf(log_survival) and log_survival < 0:
        return 1.0
    p = -math.expm1(log_survival)
    if abs(p) < EPSILON:
        return 0.0
    return clamp_probability(p)


def _effective_per_slot_from_log_survival(log_survival: float, slots: int) -> float:
    if slots <= 0:
        return 0.0
    if math.isinf(log_survival) and log_survival < 0:
        return 1.0
    p = -math.expm1(log_survival / slots)
    if abs(p) < EPSILON:
        return 0.0
    return clamp_probability(p)


def horizon_prob_with_churn(
    cfg: Config,
    target_user_seq: int,
    committee_mode: bool,
    slots: int,
    committee_size: int | None = None,
    model: SimulationModel | None = None,
) -> float:
    if model is None:
        model = SimulationModel(cfg)
    log_survival = _horizon_log_survival(
        cfg=cfg,
        model=model,
        target_user_seq=target_user_seq,
        committee_mode=committee_mode,
        slots=slots,
        committee_size=committee_size,
    )
    return _horizon_probability_from_log_survival(log_survival)


def expected_delay_hours_with_churn(
    cfg: Config,
    target_user_seq: int,
    committee_mode: bool,
    committee_size: int | None = None,
    model: SimulationModel | None = None,
) -> float:
    if model is None:
        model = SimulationModel(cfg)
    # E[T] for time-varying Bernoulli process:
    # E[T_slots] = sum_{s>=0} P(T_slots > s)
    epochs_to_full = math.ceil(target_user_seq / cfg.max_new_sequencers_per_epoch)

    survival = 1.0
    expected_slots = 0.0

    # Onboarding phase (parameters change per epoch).
    for epoch_idx in range(epochs_to_full):
        active_user = _active_user_for_epoch(cfg, target_user_seq, epoch_idx)
        f, g = _epoch_factors(
            model=model,
            active_user=active_user,
            slots_in_epoch=cfg.epoch_slots,
            committee_mode=committee_mode,
            committee_size=committee_size,
        )
        expected_slots += survival * g
        survival *= f

    # Steady state after full onboarding (same parameters every epoch).
    f_final, g_final = _epoch_factors(
        model=model,
        active_user=target_user_seq,
        slots_in_epoch=cfg.epoch_slots,
        committee_mode=committee_mode,
        committee_size=committee_size,
    )
    if f_final >= 1.0 - EPSILON:
        return float("inf")
    expected_slots += survival * g_final / (1.0 - f_final)
    return expected_slots * (cfg.slot_seconds / 3600.0)


def effective_per_slot_with_churn(
    cfg: Config,
    target_user_seq: int,
    committee_mode: bool,
    slots: int,
    committee_size: int | None = None,
    model: SimulationModel | None = None,
) -> float:
    if model is None:
        model = SimulationModel(cfg)
    log_survival = _horizon_log_survival(
        cfg=cfg,
        model=model,
        target_user_seq=target_user_seq,
        committee_mode=committee_mode,
        slots=slots,
        committee_size=committee_size,
    )
    return _effective_per_slot_from_log_survival(log_survival, slots)


def _polyline(xs: list[float], ys: list[float], x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    pts: list[str] = []
    for x, y in zip(xs, ys):
        px = 80 + (x - x_min) / (x_max - x_min) * (1100 - 80) if x_max > x_min else 80
        py = 560 - (y - y_min) / (y_max - y_min) * (560 - 40) if y_max > y_min else 560
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def _padded_bounds(values: list[float], relative_padding: float = 0.05, absolute_padding: float = 1.0) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if upper - lower > EPSILON:
        return lower, upper
    center = lower
    pad = max(abs(center) * relative_padding, absolute_padding)
    return center - pad, center + pad


def _visible_segments(xs: list[float], ys: list[float], y_min: float, y_max: float) -> list[tuple[list[float], list[float]]]:
    segments: list[tuple[list[float], list[float]]] = []
    seg_xs: list[float] = []
    seg_ys: list[float] = []
    for x, y in zip(xs, ys):
        if math.isfinite(x) and math.isfinite(y) and y_min <= y <= y_max:
            seg_xs.append(x)
            seg_ys.append(y)
            continue
        if len(seg_xs) >= 2:
            segments.append((seg_xs, seg_ys))
        seg_xs = []
        seg_ys = []
    if len(seg_xs) >= 2:
        segments.append((seg_xs, seg_ys))
    return segments


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
    if not xs:
        raise ValueError("xs must not be empty")
    x_min = min(xs) if x_min is None else x_min
    x_max = max(xs) if x_max is None else x_max
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")

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
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min")
    if x_ticks_usd_and_seq and usd_per_seq <= 0:
        raise ValueError("usd_per_seq must be > 0 when x_ticks_usd_and_seq is enabled")

    title_xml = html.escape(title)
    x_label_xml = html.escape(x_label)
    y_label_xml = html.escape(y_label)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="640">',
        '<defs>',
        '  <clipPath id="plot-area-clip">',
        '    <rect x="80" y="40" width="1020" height="520"/>',
        '  </clipPath>',
        '</defs>',
        '<rect x="0" y="0" width="1200" height="640" fill="white"/>',
        '<line x1="80" y1="560" x2="1100" y2="560" stroke="black" stroke-width="2"/>',
        '<line x1="80" y1="560" x2="80" y2="40" stroke="black" stroke-width="2"/>',
        f'<text x="600" y="24" font-family="sans-serif" font-size="20" text-anchor="middle">{title_xml}</text>',
        f'<text x="600" y="620" font-family="sans-serif" font-size="16" text-anchor="middle">{x_label_xml}</text>',
        f'<text x="20" y="300" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90,20,300)">{y_label_xml}</text>',
    ]

    for i, (name, _, color, dash) in enumerate(series):
        fxs, fys = filtered[i]
        if not fxs:
            continue
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        for seg_xs, seg_ys in _visible_segments(fxs, fys, y_min, y_max):
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} '
                f'clip-path="url(#plot-area-clip)" '
                f'points="{_polyline(seg_xs, seg_ys, x_min, x_max, y_min, y_max)}"/>'
            )
        lines.append(
            f'<text x="860" y="{70 + i * 24}" font-family="sans-serif" '
            f'font-size="14" fill="{color}">{html.escape(name)}</text>'
        )

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
    cfg.validate()
    model = SimulationModel(cfg)

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
        _, _, p_allow = model.committee_epoch_factors(user_seq=user_seq, slots_in_epoch=cfg.epoch_slots)

        log_survival_non = _horizon_log_survival(
            cfg=cfg,
            model=model,
            target_user_seq=user_seq,
            committee_mode=False,
            slots=slots,
        )
        log_survival_com = _horizon_log_survival(
            cfg=cfg,
            model=model,
            target_user_seq=user_seq,
            committee_mode=True,
            slots=slots,
        )
        p_horizon_non = _horizon_probability_from_log_survival(log_survival_non)
        p_horizon_com = _horizon_probability_from_log_survival(log_survival_com)

        rows.append(
            {
                "user_sequencers": user_seq,
                "invested_stake_token": invested_token,
                "invested_stake_usd": invested_usd,
                "p_committee_allows_honest": p_allow,
                p_horizon_non_key: p_horizon_non,
                p_horizon_com_key: p_horizon_com,
                p_eff_non_key: _effective_per_slot_from_log_survival(log_survival_non, slots),
                p_eff_com_key: _effective_per_slot_from_log_survival(log_survival_com, slots),
                "expected_hours_non_committee": expected_delay_hours_with_churn(
                    cfg=cfg,
                    model=model,
                    target_user_seq=user_seq,
                    committee_mode=False,
                ),
                "expected_hours_committee": expected_delay_hours_with_churn(
                    cfg=cfg,
                    model=model,
                    target_user_seq=user_seq,
                    committee_mode=True,
                ),
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
        path=figures_dir / "cr_probability_vs_stake.svg",
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

    finite_delay_values = [
        v
        for r in rows
        for v in (r["expected_hours_non_committee"], r["expected_hours_committee"])
        if math.isfinite(v)
    ]
    delay_y_cap_hours = 30.0 * 24.0
    delay_x_min, delay_x_max = _padded_bounds(invested, absolute_padding=usd_per_seq)
    if finite_delay_values:
        delay_y_min, delay_y_max = _padded_bounds(
            finite_delay_values,
            absolute_padding=cfg.slot_seconds / 3600.0,
        )
        delay_y_max = min(delay_y_max, delay_y_cap_hours)
        if delay_y_max <= delay_y_min:
            delay_y_min = 0.0
            delay_y_max = delay_y_cap_hours
    else:
        delay_y_min, delay_y_max = 0.0, delay_y_cap_hours

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
        x_min=delay_x_min,
        x_max=delay_x_max,
        y_min=delay_y_min,
        y_max=delay_y_max,
        y_tick_decimals=2,
        x_ticks_usd_and_seq=True,
        usd_per_seq=usd_per_seq,
    )

    print(f"Wrote {main_csv}")
    print("Wrote figures:")
    print(" - figures/cr_probability_vs_stake.svg")
    print(" - figures/cr_per_slot_probability_vs_stake.svg")
    print(" - figures/cr_expected_delay_vs_stake.svg")
    print(f"Slots in horizon: {slots}")


if __name__ == "__main__":
    run(Config())
