"""Run the selector on generated domain labels, without downloading data."""

from mixture_then_select import select


domains = ["math", "code", "dialog"]
pool = ["math"] * 40 + ["code"] * 35 + ["dialog"] * 25
result = select(
    pool,
    {"math": 2, "code": 4, "dialog": 14},
    budget=20,
    domain_order=domains,
)
print(result.to_dict())
