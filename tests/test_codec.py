from dataclasses import replace

import pytest

from lec.codec import LECCodec
from lec.config import LECConfig
from lec.crypto import CryptoError
from lec.model import ToyLanguageModel
from lec.probabilities import SamplingPolicy
from lec.range_coder import CodingError


@pytest.fixture()
def codec() -> LECCodec:
    config = LECConfig(
        capacity=8,
        max_tokens=1000,
        policy=SamplingPolicy(temperature=0.9, top_k=16, precision_bits=12),
    )
    return LECCodec(config, ToyLanguageModel())


@pytest.mark.parametrize("message", ["", "hello", "café", "🙂"])
def test_roundtrip(codec: LECCodec, message: str) -> None:
    result = codec.encode(message, "correct horse", nonce=bytes(range(12)))
    recovered, metrics = codec.decode(result.text, "correct horse")
    assert recovered == message
    assert metrics.tokens == result.metrics.tokens
    assert result.metrics.payload_bits == codec.config.payload_bits


def test_ciphertext_nonce_changes_covertext(codec: LECCodec) -> None:
    first = codec.encode("hello", "key", nonce=b"\x00" * 12)
    second = codec.encode("hello", "key", nonce=b"\x01" * 12)
    assert first.text != second.text


def test_wrong_key_is_authenticated(codec: LECCodec) -> None:
    result = codec.encode("hello", "right", nonce=b"\x02" * 12)
    with pytest.raises(CryptoError, match="authentication failed"):
        codec.decode(result.text, "wrong")


def test_modified_covertext_fails(codec: LECCodec) -> None:
    result = codec.encode("hello", "key", nonce=b"\x03" * 12)
    tokens = result.text.split()
    tokens[-1] = "quiet" if tokens[-1] != "quiet" else "train"
    with pytest.raises((CodingError, CryptoError)):
        codec.decode(" ".join(tokens), "key")


def test_capacity_is_part_of_configuration(codec: LECCodec) -> None:
    result = codec.encode("hello", "key", nonce=b"\x04" * 12)
    other = LECCodec(replace(codec.config, capacity=9), ToyLanguageModel())
    with pytest.raises((CodingError, CryptoError)):
        other.decode(result.text, "key")


def test_oversized_message(codec: LECCodec) -> None:
    with pytest.raises(CryptoError, match="capacity"):
        codec.encode("more than eight bytes", "key")


def test_encode_reports_live_token_snapshots(codec: LECCodec) -> None:
    snapshots: list[tuple[str, int]] = []
    result = codec.encode(
        "stream",
        "key",
        nonce=b"\x05" * 12,
        on_token=lambda text, count: snapshots.append((text, count)),
    )
    assert len(snapshots) == result.metrics.tokens
    assert snapshots[-1] == (result.text, result.metrics.tokens)
