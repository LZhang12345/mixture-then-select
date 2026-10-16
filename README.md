# Mixture-then-Select

**Choose how much data to take from each domain, then choose the samples.**

Code for [Pick the Mixture, Not the Samples: Combinatorial Data Selection for LLMs](https://arxiv.org/abs/2610.14028) by Luyang Zhang, Jialu Wang, and Bebei Li.

## Intuition

A sample's usefulness depends on what else is selected: some samples reinforce one another, while others repeat information. Ranking every sample in one list can therefore miss useful combinations across domains. Mixture-then-Select separates two decisions:

1. **Choose the mixture.** Use calibration measurements to predict which domain counts will be useful.
2. **Fill the allocation.** Select the requested number of samples from each domain.

This repository provides the CPU allocation and selection code for Mixture-then-Select (MtS).

## Mathematical intuition

The paper approximates a subset's utility using individual values and pairwise interactions:

$$
u(S) \approx \sum_{i\in S} v_i + \sum_{\{i,j\}\subseteq S}\varphi_{ij}.
$$

Here, $v_i$ measures sample $i$'s individual value. A positive $\varphi_{ij}$ means that the pair is more useful together; a negative value indicates redundancy or interference. Ranking only by $v_i$ leaves these interactions out.

Now let $n_c$ be the number of selected samples from domain $c$. If $\bar\varphi_{cd}$ is the mean interaction between domains $c$ and $d$, their contribution is approximately $\bar\varphi_{cd}n_cn_d$. This gives the domain counts a direct role in utility and motivates learning a predictor $\hat g$ over allocations:

$$
\hat{\mathbf n}\in\arg\max_{\mathbf n}\hat g(\mathbf n)
\quad\text{subject to}\quad
\sum_c n_c=m,\qquad 0\le n_c\le |C_c|,\qquad n_c\in\mathbb Z.
$$

Here, $m$ is the sample budget and $C_c$ is the pool of samples in domain $c$. Once the counts are chosen, selection fills each domain's allocation. The search is over domain counts, while within-domain selection determines the individual samples.

## Default allocation and selection

The implementation blends the learned allocation equally with the pool-proportional allocation:

$$
\mathbf n^{\mathrm{pool}}=\operatorname{Round}_m\!\left(m\frac{(|C_1|,\ldots,|C_T|)}{N}\right),
\qquad
\mathbf n=\operatorname{Round}_m\!\left(\frac{\hat{\mathbf n}+\mathbf n^{\mathrm{pool}}}{2}\right).
$$

$N$ is the pool size. $\operatorname{Round}_m$ uses largest-remainder rounding so the integer counts sum to exactly $m$. The blend keeps part of the pool's original composition while moving toward the learned allocation. The code then fills each domain using a deterministic random order with seed 42.

For example, if the pool proportions assign 8 slots to a domain and the learned allocation assigns 2, the blended allocation assigns 5. All domain counts together still sum to the selection budget.

The paper's Algorithm 1 fills domains by attribution ranking; the default implementation here uses deterministic random completion within each domain.

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

`learned_counts` is the allocation produced by your calibrated utility predictor, before blending with the pool proportions. It must sum to the requested budget and fit within each domain's capacity. Calibration measurements and model training are supplied by the caller.

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

The release was also replayed against the original 50K domain metadata: all 1,000 selected indices match the archived selection exactly.

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
