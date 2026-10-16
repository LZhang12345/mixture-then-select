"""Select pool positions from a local JSON allocation specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import select


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select samples with Mixture-then-Select")
    parser.add_argument("--input", required=True, type=Path, help="local JSON specification")
    parser.add_argument("--output", required=True, type=Path, help="new selection JSON file")
    args = parser.parse_args()
    try:
        if args.output.exists() and args.input.resolve() == args.output.resolve():
            raise ValueError("input and output must be different files")
        with args.input.open(encoding="utf-8") as stream:
            document = json.load(stream, object_pairs_hook=_unique_keys)
        if not isinstance(document, dict):
            raise ValueError("input must be a JSON object")
        required = {"sample_domains", "learned_counts", "budget", "domain_order"}
        allowed = required | {"seed"}
        if required - document.keys() or document.keys() - allowed:
            raise ValueError("input fields must be sample_domains, learned_counts, budget, domain_order and optional seed")
        if not isinstance(document["sample_domains"], list):
            raise ValueError("sample_domains must be a JSON list")
        if not isinstance(document["learned_counts"], dict):
            raise ValueError("learned_counts must be a JSON object")
        if not isinstance(document["domain_order"], list):
            raise ValueError("domain_order must be a JSON list")
        result = select(**document)
        payload = json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Selected {len(result.indices)} samples; SHA-256 {result.to_dict()['selection_sha256']}")


if __name__ == "__main__":
    main()
