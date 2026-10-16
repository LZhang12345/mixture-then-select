<div align="center">

<h1>Mixture-then-Select</h1>

<p><strong>Pick the Mixture, Not the Samples:<br>Combinatorial Data Selection for LLMs</strong></p>

<p>Luyang Zhang &middot; Jialu Wang &middot; Bebei Li</p>

<p>
  <a href="https://arxiv.org/abs/2610.14028"><img src="https://img.shields.io/badge/arXiv-2610.14028-b31b1b" alt="Paper on arXiv"></a>
  <a href="https://github.com/LZhang12345/mixture-then-select/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/LZhang12345/mixture-then-select/tests.yml?branch=main&amp;label=tests" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-555555" alt="MIT license"></a>
</p>

<p><a href="#quick-start">Quick start</a> &middot; <a href="#method">Method</a> &middot; <a href="#use-your-own-data">Usage</a> &middot; <a href="#citation">Citation</a></p>

</div>

## Overview

**Choose how much data to take from each domain, then choose the samples.**

A sample's usefulness depends on what else is selected: some samples reinforce one another, while others repeat information. Ranking every sample in one list can therefore miss useful combinations across domains. Mixture-then-Select (MtS) separates two decisions:

1. **Choose the mixture.** Use calibration measurements to predict which domain counts will be useful.
2. **Fill the allocation.** Select the requested number of samples from each domain.

This repository provides the CPU allocation and selection code. It takes a learned domain allocation as input.

## Quick start

Requires **Python 3.10+** and **NumPy**. The example runs on CPU and creates its inputs in memory.

```bash
git clone https://github.com/LZhang12345/mixture-then-select.git
cd mixture-then-select
python -m pip install -e .
python examples/toy_selection.py
```

## Method

The paper approximates a subset's utility using individual values and pairwise interactions:

```math
u(S) \approx \sum_{i\in S} v_i + \sum_{i<j,\ i,j\in S}\varphi_{ij}.
```

Here, $`v_i`$ is sample $`i`$'s individual value. Positive $`\varphi_{ij}`$ indicates complementarity; negative values indicate redundancy or interference. Ranking only by individual values leaves these interactions out.

If $`n_c`$ samples come from domain $`c`$, the mean interaction between domains $`c`$ and $`d`$ contributes approximately $`\bar{\varphi}_{cd}n_cn_d`$. Domain counts therefore affect utility directly. MtS uses a learned predictor $`\hat g`$ to choose those counts:

```math
\hat{\mathbf n}\in\arg\max_{\mathbf n\in\mathcal F_m}\hat g(\mathbf n).
```

The feasible set $`\mathcal F_m`$ contains integer domain counts that sum to the sample budget $`m`$ without exceeding any domain's capacity. Once the counts are chosen, selection fills each domain's allocation.

**Implementation.** The code blends the learned allocation equally with the pool-proportional allocation, then selects samples using a deterministic random order within each domain. Algorithm 1 in the paper uses attribution ranking for this final step.

## Use your own data

Supply your ordered pool labels and learned allocation:

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

The output contains selected pool positions, domain counts, the seed, and a selection hash. Choose a new output filename for each run; existing files are preserved.

### Connect your predictor

`learned_counts` is your predictor's allocation before blending. It must sum to `budget` and fit within each domain's capacity. Calibration measurements and model training are supplied by the caller.

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

The candidate allocations and scores above are illustrative.

<details>
<summary><strong>Allocation and reproducibility details</strong></summary>

Let $`p_c=|C_c|/N`$ be domain $`c`$'s share of the pool. The implementation first rounds the pool-proportional allocation, then rounds the equal blend with the learned allocation:

$`\mathbf n^{\mathrm{pool}}=R_m(m\mathbf p).`$

$`\mathbf n=R_m\left(\frac{\hat{\mathbf n}+\mathbf n^{\mathrm{pool}}}{2}\right).`$

$`R_m`$ uses largest-remainder rounding to preserve the exact budget. Ties follow `domain_order`. For example, a domain assigned 8 slots by pool proportions and 2 by the learned allocation receives 5 before final rounding.

Selection is deterministic for a fixed pool order, domain order, learned allocation, and selection seed. The default selection seed is 42; later training seeds can vary independently.

</details>

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover exact quotas, capacity limits, deterministic selection, nested per-domain ordering, malformed inputs, and the frozen eight-domain allocation.

The release was also replayed against the original 50K domain metadata: all 1,000 selected indices match the archived selection exactly.

## Citation

If you use this code, please cite the paper:

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

## License

MIT. See [LICENSE](LICENSE).
