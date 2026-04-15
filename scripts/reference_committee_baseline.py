#!/usr/bin/env python3

from math import ceil, comb, log

N = 4000
n = 48
censoring = 0.50  # largest censoring minority (honest majority assumption for CR)
epoch_slots = 32
slot_seconds = 72
target_survival = 0.01  # 99% inclusion

K = int(censoring * N)
max_censors_allowed = (n - 1) // 3  # 15, so 16 censors in a committee block
denom = comb(N, n)

# Committee members are sampled without replacement, so censor counts are hypergeometric.
partial_survival = [0.0] * (epoch_slots + 1)
partial_survival[0] = 1.0
for x in range(n + 1):
    probability = comb(K, x) * comb(N - K, n - x) / denom
    if x > max_censors_allowed:
        for slot in range(1, epoch_slots + 1):
            partial_survival[slot] += probability
        continue
    miss = x / n
    miss_power = 1.0
    for slot in range(1, epoch_slots + 1):
        miss_power *= miss
        partial_survival[slot] += probability * miss_power

p_epoch_censored = partial_survival[epoch_slots]
full_epochs_before_search = max(0, ceil(log(target_survival) / log(p_epoch_censored)) - 1)
log_prefix_survival = 0.0 if full_epochs_before_search == 0 else full_epochs_before_search * log(p_epoch_censored)

for slot in range(1, epoch_slots + 1):
    if log_prefix_survival + log(partial_survival[slot]) <= log(target_survival) + 1e-12:
        target_slots = full_epochs_before_search * epoch_slots + slot
        break
else:
    target_slots = (full_epochs_before_search + 1) * epoch_slots

print(f"p_epoch_censored= {p_epoch_censored:.12f}")
print(f"slots_to_99pct= {target_slots}")
print(f"days_to_99pct= {target_slots * slot_seconds / 86400:.6f}")
