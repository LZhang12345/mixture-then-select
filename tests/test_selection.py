from collections import Counter
import hashlib
import json
import random
import unittest

import numpy as np

from mixture_then_select import allocate, choose_mixture, select, selection_hash


DOMAINS = (
    "math", "code", "general_qa", "dialog", "reasoning", "nlp_tasks",
    "complex_instructions", "creative",
)


class SelectionTests(unittest.TestCase):
    def test_frozen_eight_domain_allocation(self):
        pool = dict(zip(DOMAINS, [6500] * 4 + [6000] * 4))
        learned = dict(zip(DOMAINS, [5, 117, 193, 300, 96, 3, 65, 221]))
        result = allocate(pool, learned, budget=1000, domain_order=DOMAINS)
        self.assertEqual(list(result.values()), [68, 124, 162, 215, 108, 61, 92, 170])

    def test_exact_budget_and_observed_counts(self):
        labels = ["a", "b", "c"] * 50
        result = select(labels, {"a": 4, "b": 7, "c": 10}, budget=21, domain_order=["a", "b", "c"])
        self.assertEqual(len(result.indices), 21)
        self.assertEqual(len(set(result.indices)), 21)
        self.assertEqual(tuple(sorted(result.indices)), result.indices)
        observed = Counter(labels[index] for index in result.indices)
        self.assertEqual(dict(observed), dict(zip(result.domain_order, result.counts)))

    def test_repeated_call_is_identical(self):
        labels = ["a", "b"] * 50
        kwargs = dict(budget=20, domain_order=["a", "b"], seed=42)
        first = select(labels, {"a": 8, "b": 12}, **kwargs)
        random.seed(999)
        np.random.seed(777)
        self.assertEqual(first, select(labels, {"a": 8, "b": 12}, **kwargs))

    def test_random_state_is_not_modified(self):
        random.seed(617)
        before = random.getstate()
        select(["a"] * 30, {"a": 10}, budget=10, domain_order=["a"])
        self.assertEqual(before, random.getstate())

    def test_seeds_change_selection(self):
        kwargs = dict(budget=10, domain_order=["a"])
        first = select(["a"] * 100, {"a": 10}, seed=42, **kwargs)
        second = select(["a"] * 100, {"a": 10}, seed=123, **kwargs)
        self.assertNotEqual(first.indices, second.indices)

    def test_per_domain_orders_are_nested(self):
        labels = ["a", "b"] * 100
        first = select(labels, {"a": 10, "b": 30}, budget=40, domain_order=["a", "b"])
        second = select(labels, {"a": 30, "b": 10}, budget=40, domain_order=["a", "b"])
        in_domain = lambda result, domain: {i for i in result.indices if labels[i] == domain}
        self.assertLess(in_domain(first, "a"), in_domain(second, "a"))
        self.assertGreater(in_domain(first, "b"), in_domain(second, "b"))

    def test_whole_pool_and_empty_domain(self):
        result = select(["a"] * 4, {"a": 4, "empty": 0}, budget=4, domain_order=["a", "empty"])
        self.assertEqual(result.indices, (0, 1, 2, 3))
        self.assertEqual(result.counts, (4, 0))

    def test_rounding_tie_follows_domain_order(self):
        pool = {"a": 1, "b": 1}
        learned = {"a": 0, "b": 1}
        self.assertEqual(allocate(pool, learned, budget=1, domain_order=["a", "b"]), {"a": 1, "b": 0})

    def test_choose_best_candidate_and_first_tie(self):
        candidates = [{"a": 1, "b": 3}, {"a": 3, "b": 1}]
        kwargs = dict(pool_counts={"a": 5, "b": 5}, budget=4, domain_order=["a", "b"])
        self.assertEqual(choose_mixture(candidates, [0.2, 0.3], **kwargs), candidates[1])
        self.assertEqual(choose_mixture(candidates, [0.3, 0.3], **kwargs), candidates[0])

    def test_invalid_predictions_and_candidates(self):
        kwargs = dict(pool_counts={"a": 5}, budget=2, domain_order=["a"])
        for candidates, scores in [([], []), ([{"a": 2}], []), ([{"a": 2}], [float("nan")]), ([{"a": 6}], [1.0]), ([{"a": 1}], [1.0])]:
            with self.subTest(candidates=candidates, scores=scores):
                with self.assertRaises(ValueError):
                    choose_mixture(candidates, scores, **kwargs)

    def test_reject_unknown_domain(self):
        with self.assertRaises(ValueError):
            select(["a", "missing"], {"a": 1}, budget=1, domain_order=["a"])

    def test_reject_invalid_budgets(self):
        for budget in [0, -1, 1.5, True, 101]:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    select(["a"] * 100, {"a": 1}, budget=budget, domain_order=["a"])

    def test_reject_invalid_counts_and_domain_orders(self):
        for counts, order in [({"a": 2.0}, ["a"]), ({"a": -2}, ["a"]), ({"a": True}, ["a"]), ({"a": 1}, ["a"]), ({"a": 2, "b": 0}, ["a"]), ({"a": 2}, ["a", "a"]), ({"a": 2}, "a")]:
            with self.subTest(counts=counts, order=order):
                with self.assertRaises(ValueError):
                    select(["a"] * 10, counts, budget=2, domain_order=order)

    def test_reject_excess_domain_capacity(self):
        with self.assertRaises(ValueError):
            select(["a", "b", "b"], {"a": 2, "b": 0}, budget=2, domain_order=["a", "b"])

    def test_hash_is_canonical(self):
        expected = hashlib.sha256(b"[1,4,9]").hexdigest()
        self.assertEqual(selection_hash([9, 1, 4]), expected)
        with self.assertRaises(ValueError):
            selection_hash([1, 1])

    def test_output_has_selection_metadata(self):
        result = select(["a"] * 4, {"a": 2}, budget=2, domain_order=["a"])
        encoded = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(encoded["method"], "mixture_then_select")
        self.assertEqual(encoded["allocation_weight"], 0.5)
        self.assertEqual(encoded["selection_sha256"], selection_hash(result.indices))


if __name__ == "__main__":
    unittest.main()
