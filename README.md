# Mixture-then-Select

**Choose how much data to take from each domain, then choose the samples.**

Code for [Pick the Mixture, Not the Samples: Combinatorial Data Selection for LLMs](https://arxiv.org/abs/2610.14028) by Luyang Zhang, Jialu Wang, and Bebei Li.

## Intuition

A good training subset needs a useful combination of skills. Ranking every sample in one list can spend most of the budget on a few domains. Mixture-then-Select makes the domain allocation an explicit decision:

1. **Choose the mixture.** Use calibration measurements to predict which domain counts will be useful.
2. **Fill the allocation.** Select the requested number of samples from each domain.

The paper develops this separation and analyzes interactions between training samples. This release provides the CPU selection component of **MtS-lambda0.50**, the strongest three-seed configuration in our local fine-lambda and selection-engineering search.

## Released configuration

MtS-lambda0.50 takes an already learned allocation and blends it equally with the pool-proportional allocation. It rounds the blended counts to the exact budget and fills each domain using a deterministic random order with seed 42.

For example, if the pool proportions assign 8 slots to a domain and the learned allocation assigns 2, the blended allocation assigns 5. All domain counts together still sum to the selection budget.

The paper's Algorithm 1 fills domains by attribution ranking; this released variant uses deterministic random completion within each domain.

## Install and run

Python 3.10 or newer and NumPy are sufficient. Selection runs on CPU.

```bash
git clone https://github.com/LZhang12345/mixture-then-select.git
cd mixture-then-select
python -m pip install -e .
python examples/toy_selection.py
```

The example creates domain labels in memory. Supply your own ordered pool labels and learned allocation for real use:

```python
from mixture_then_select import select

domain_order = ["math", "code", "dialog"]
sample_domains = ["math"] * 40 + ["code"] * 35 + ["dialog"] * 25

result = select(
    sample_domains,
    learned_counts={"math": 2, "code": 4, "dialog": 14},
    budget=20,
    domain_order=domain_order,
    seed=42,
)

print(dict(zip(result.domain_order, result.counts)))
# {'math': 5, 'code': 6, 'dialog': 9}
print(result.indices)  # zero-based positions in your input pool
```

The command-line interface accepts the same inputs as JSON:

```bash
mts-select --input examples/toy_input.json --output selection.json
```

The output includes selected positions, domain counts, the selection seed, and a SHA-256 of the selected index list. Existing output files are preserved; choose a new output filename for another run.

## Connecting Stage 1

`learned_counts` is the allocation produced by your calibrated utility predictor, before the lambda0.50 blend. It must sum to the requested budget and fit within each domain's capacity. Calibration measurements, model training, and attribution scores are supplied by the caller.

If your predictor scores a finite catalog of feasible allocations, `choose_mixture` selects the highest-scoring candidate:

```python
from mixture_then_select import choose_mixture

learned_counts = choose_mixture(
    [{"math": 2, "code": 4, "dialog": 14},
     {"math": 6, "code": 6, "dialog": 8}],
    predicted_utilities=[0.72, 0.68],  # your predictor's outputs
    pool_counts={"math": 40, "code": 35, "dialog": 25},
    budget=20,
    domain_order=["math", "code", "dialog"],
)
```

The candidate catalog and predictor outputs above are illustrative. Selection is deterministic for a fixed pool order, domain order, learned allocation, and selection seed. Training seeds can vary independently. In the released 50K rule, pool-proportional counts are rounded before blending and allocation ties follow `domain_order`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover exact quotas, capacity limits, deterministic selection, nested per-domain ordering, malformed inputs, and the frozen eight-domain allocation.

The release was also replayed against the original 50K domain metadata: all 1,000 selected indices match the archived lambda0.50 selection exactly.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{zhang2026pick,
  title         = {Pick the Mixture, Not the Samples: Combinatorial Data Selection for {LLMs}},
  author        = {Luyang Zhang and Jialu Wang and Bebei Li},
  year          = {2026},
  eprint        = {2610.14028},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.14028}
}
```
