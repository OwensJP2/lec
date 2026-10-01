"""Run a deterministic offline reproducibility check."""

from lec.codec import LECCodec
from lec.config import LECConfig
from lec.model import ToyLanguageModel
from lec.probabilities import SamplingPolicy


def main() -> None:
    config = LECConfig(
        capacity=16,
        policy=SamplingPolicy(top_k=24, precision_bits=16),
    )
    codec = LECCodec(config, ToyLanguageModel())
    nonce = bytes.fromhex("000102030405060708090a0b")
    first = codec.encode("reproducible", "test key", nonce=nonce)
    second = codec.encode("reproducible", "test key", nonce=nonce)
    recovered, _ = codec.decode(first.text, "test key")
    assert first.text == second.text
    assert recovered == "reproducible"
    print("PASS")
    print(f"tokens={first.metrics.tokens}")
    print(f"bits_per_token={first.metrics.bits_per_token:.4f}")


if __name__ == "__main__":
    main()
