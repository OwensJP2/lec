"""Collect recovery and capacity measurements as JSON Lines."""

from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import asdict
from pathlib import Path

from lec.codec import LECCodec
from lec.config import LECConfig
from lec.model import ToyLanguageModel
from lec.probabilities import SamplingPolicy


def message(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_letters + string.digits + " .,!?-"
    return "".join(rng.choice(alphabet) for _ in range(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 0.9, 1.1])
    parser.add_argument("--top-k", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--precision-bits", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = ToyLanguageModel() if args.toy else None
    with args.output.open("w", encoding="utf-8") as output:
        for size in args.sizes:
            for temperature in args.temperatures:
                for top_k in args.top_k:
                    for precision in args.precision_bits:
                        policy = SamplingPolicy(
                            temperature=temperature,
                            top_k=top_k,
                            precision_bits=precision,
                        )
                        config = LECConfig(capacity=size, policy=policy)
                        codec = LECCodec(config, model=model)
                        for sample in range(args.samples):
                            plaintext = message(rng, rng.randint(0, size))
                            encoded = codec.encode(plaintext, "experiment-key")
                            recovered, _ = codec.decode(encoded.text, "experiment-key")
                            record = {
                                "sample": sample,
                                "message_bytes": len(plaintext.encode()),
                                "recovered": recovered == plaintext,
                                **asdict(policy),
                                **encoded.metrics.to_dict(),
                            }
                            output.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
