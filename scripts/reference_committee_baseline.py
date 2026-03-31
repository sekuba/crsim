#!/usr/bin/env python3

from math import ceil, comb, log

N = 4000
n = 48
censoring = 0.50 # largest censoring minority (honest majority assumption for CR)
epoch_slots = 32
slot_seconds = 72
target_survival = 0.01  # 99% inclusion

K = int(censoring * N)
max_censors_allowed = (n - 1) // 3  # 15, so 16 censors in a committee block
denom = comb(N, n)
# Committee members are sampled without replacement, so censor counts are hypergeometric.
probability_mass_function = lambda x: comb(K, x) * comb(N - K, n - x) / denom

p_epoch_censored = sum(
    probability_mass_function(x) if x > max_censors_allowed else probability_mass_function(x) * (x / n) ** epoch_slots
    for x in range(n + 1)
)
epochs = ceil(log(target_survival) / log(p_epoch_censored))
days = epochs * epoch_slots * slot_seconds / 86400

print(f"p_epoch_censored= {p_epoch_censored:.12f}")
print(f"epochs_to_99pct= {epochs}")
print(f"days_to_99pct= {days:.6f}")
