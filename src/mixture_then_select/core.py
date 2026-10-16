"""Mixture-then-Select with equal learned and pool-proportional allocation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np


ALLOCATION_WEIGHT = 0.50


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _domains(domain_order: Sequence[str]) -> tuple[str, ...]:
    if isinstance(domain_order, (str, bytes)):
        raise ValueError("domain_order must be a sequence of domain names")
    names = tuple(domain_order)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("domain names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("domain_order must contain each domain exactly once")
    return names


def _counts(
    values: Mapping[str, int], domain_order: tuple[str, ...], name: str
) -> np.ndarray:
    if set(values) != set(domain_order):
        raise ValueError(f"{name} must contain exactly the domains in domain_order")
    counts = [_integer(values[domain], f"{name}[{domain}]") for domain in domain_order]
    if sum(counts) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported integer range")
    return np.asarray(counts, dtype=np.int64)


def _round_counts(values: np.ndarray, total: int) -> np.ndarray:
    """Largest remainders; ties go to the earlier domain in domain_order."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("allocation weights must be a finite nonnegative vector")
    weight_sum = float(values.sum())
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        raise ValueError("allocation weights must have a finite positive sum")
    scaled = values * (total / weight_sum)
    counts = np.floor(scaled).astype(np.int64)
    deficit = total - int(counts.sum())
    if not 0 <= deficit <= len(values):
        raise ValueError("allocation rounding exceeded floating-point precision")
    order = np.lexsort((np.arange(values.size), -(scaled - counts)))
    counts[order[:deficit]] += 1
    if int(counts.sum()) != total:
        raise RuntimeError("allocation does not match the requested budget")
    return counts


def allocate(
    pool_counts: Mapping[str, int],
    learned_counts: Mapping[str, int],
    *,
    budget: int,
    domain_order: Sequence[str],
) -> dict[str, int]:
    """Blend learned counts and pool-proportional counts equally, then round.

    learned_counts must be a feasible learned allocation at the requested
    budget. Pool-proportional counts are rounded before blending, as in the
    evaluated 50K implementation.
    """
    budget = _integer(budget, "budget", minimum=1)
    names = _domains(domain_order)
    capacity = _counts(pool_counts, names, "pool_counts")
    learned = _counts(learned_counts, names, "learned_counts")
    if budget > int(capacity.sum()):
        raise ValueError("budget exceeds the available pool")
    if int(learned.sum()) != budget:
        raise ValueError("learned_counts must sum to budget")
    if np.any(learned > capacity):
        raise ValueError("learned_counts exceed a domain's available samples")
    proportional = _round_counts(capacity, budget)
    blended = (1.0 - ALLOCATION_WEIGHT) * proportional + ALLOCATION_WEIGHT * learned
    counts = _round_counts(blended, budget)
    if np.any(counts > capacity):
        raise RuntimeError("rounded allocation exceeds a domain's capacity")
    return {domain: int(counts[index]) for index, domain in enumerate(names)}


def choose_mixture(
    candidate_counts: Sequence[Mapping[str, int]],
    predicted_utilities: Sequence[float],
    *,
    pool_counts: Mapping[str, int],
    budget: int,
    domain_order: Sequence[str],
) -> dict[str, int]:
    """Choose the highest-scoring feasible candidate from a calibrated predictor.

    Candidate utilities are supplied by the caller's predictor. Ties select
    the first candidate. This finite-candidate search does not claim a global
    optimum over allocations absent from candidate_counts.
    """
    budget = _integer(budget, "budget", minimum=1)
    names = _domains(domain_order)
    capacity = _counts(pool_counts, names, "pool_counts")
    if budget > int(capacity.sum()):
        raise ValueError("budget exceeds the available pool")
    candidates = list(candidate_counts)
    scores = np.asarray(predicted_utilities, dtype=np.float64)
    if not candidates or scores.shape != (len(candidates),):
        raise ValueError("supply one predicted utility per candidate")
    if not np.all(np.isfinite(scores)):
        raise ValueError("predicted utilities must be finite")
    checked = [_counts(row, names, "candidate_counts") for row in candidates]
    if any(int(row.sum()) != budget or np.any(row > capacity) for row in checked):
        raise ValueError("all candidate allocations must satisfy budget and capacity")
    winner = checked[int(np.argmax(scores))]
    return {domain: int(winner[index]) for index, domain in enumerate(names)}


def selection_hash(indices: Sequence[int]) -> str:
    """SHA-256 of the sorted, compact JSON index list."""
    checked = [_integer(index, "sample index") for index in indices]
    if len(checked) != len(set(checked)):
        raise ValueError("selection indices must be unique")
    payload = json.dumps(sorted(checked), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Selection:
    """Selected zero-based positions and allocations, in the declared order."""

    indices: tuple[int, ...]
    domain_order: tuple[str, ...]
    counts: tuple[int, ...]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "mixture_then_select",
            "allocation_weight": ALLOCATION_WEIGHT,
            "within_domain_selection": "deterministic_random",
            "seed": self.seed,
            "budget": len(self.indices),
            "domain_order": list(self.domain_order),
            "domain_counts": dict(zip(self.domain_order, self.counts)),
            "selected_indices": list(self.indices),
            "selection_sha256": selection_hash(self.indices),
        }


def select(
    sample_domains: Sequence[str],
    learned_counts: Mapping[str, int],
    *,
    budget: int,
    domain_order: Sequence[str],
    seed: int = 42,
) -> Selection:
    """Select exactly budget distinct pool positions using Mixture-then-Select.

    Input row order, domain names/order, learned counts and seed determine
    the subset. The selection seed is independent of a later training seed.
    """
    names = _domains(domain_order)
    seed = _integer(seed, "seed")
    if isinstance(sample_domains, (str, bytes)):
        raise ValueError("sample_domains must give one domain per pool row")
    pools: dict[str, list[int]] = {domain: [] for domain in names}
    for index, domain in enumerate(sample_domains):
        if not isinstance(domain, str) or domain not in pools:
            raise ValueError(f"unknown domain at sample position {index}")
        pools[domain].append(index)
    allocation = allocate(
        {domain: len(pools[domain]) for domain in names},
        learned_counts,
        budget=budget,
        domain_order=names,
    )
    selected: list[int] = []
    for domain in names:
        # Retain the namespace used by the evaluated allocation-path code.
        material = f"submitted-allocation-path:{seed}:{domain}".encode("utf-8")
        domain_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        order = list(pools[domain])
        random.Random(domain_seed).shuffle(order)
        selected.extend(order[:allocation[domain]])
    selected.sort()
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("selection does not match the requested budget")
    observed = Counter(sample_domains[index] for index in selected)
    if any(observed[domain] != allocation[domain] for domain in names):
        raise RuntimeError("selection does not match the domain allocation")
    return Selection(
        tuple(selected), names, tuple(allocation[domain] for domain in names), seed
    )
