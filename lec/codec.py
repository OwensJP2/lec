"""End-to-end linguistic entropy encoding and decoding."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .config import LECConfig
from .crypto import CryptoError, decrypt, derive_key, encrypt
from .model import LanguageModel, ModelSession, TransformersModel
from .probabilities import FrequencyTable, adjust_logits, quantize_logits
from .range_coder import CodingError, Interval, bytes_to_point, point_to_bytes


@dataclass(frozen=True)
class CodecMetrics:
    tokens: int
    payload_bits: int
    bits_per_token: float
    model_entropy_bits: float
    entropy_utilization: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EncodeResult:
    text: str
    metrics: CodecMetrics


class LECCodec:
    def __init__(
        self,
        config: LECConfig | None = None,
        model: LanguageModel | None = None,
    ) -> None:
        self.config = config or LECConfig()
        self.config.validate()
        self.model = model

    def _model(self) -> LanguageModel:
        if self.model is None:
            self.model = TransformersModel(
                self.config.model_id,
                self.config.model_revision,
            )
        return self.model

    def encode(
        self,
        message: str,
        passphrase: str,
        *,
        nonce: bytes | None = None,
        on_token: Callable[[str, int], None] | None = None,
    ) -> EncodeResult:
        plaintext = message.encode("utf-8")
        key = derive_key(passphrase)
        blob = encrypt(plaintext, key, self.config.capacity, nonce=nonce).data
        value = bytes_to_point(blob)
        interval = Interval.for_bits(len(blob) * 8)
        session = self._model().start(self.config.prompt)
        entropy = 0.0

        while not interval.complete:
            if len(session.token_ids) >= self.config.max_tokens:
                raise CodingError(
                    "max token limit reached before payload interval became unique"
                )
            table = self._table(session)
            previous_width = interval.width
            token_id = interval.choose(value, table)
            entropy += self._entropy(table)
            session.append(token_id)
            if on_token is not None:
                on_token(self._model().render(session.token_ids), len(session.token_ids))
            if interval.width == previous_width:
                raise CodingError("arithmetic interval failed to make progress")

        text = self._model().render(session.token_ids)
        if self._model().parse(text) != session.token_ids:
            raise CodingError("generated token sequence is not text-roundtrip safe")
        return EncodeResult(
            text=text,
            metrics=self._metrics(len(session.token_ids), len(blob) * 8, entropy),
        )

    def decode(self, text: str, passphrase: str) -> tuple[str, CodecMetrics]:
        token_ids = self._model().parse(text)
        if not token_ids:
            raise CodingError("covertext is empty")
        session = self._model().start(self.config.prompt)
        interval = Interval.for_bits(self.config.payload_bits)
        entropy = 0.0

        for index, token_id in enumerate(token_ids):
            if interval.complete:
                raise CodingError("covertext contains trailing tokens")
            table = self._table(session)
            interval.consume(token_id, table)
            entropy += self._entropy(table)
            session.append(token_id)
            if interval.complete and index != len(token_ids) - 1:
                raise CodingError("covertext contains trailing tokens")

        if not interval.complete:
            raise CodingError("covertext is truncated or uses a different configuration")
        blob = point_to_bytes(interval.low, self.config.payload_bytes)
        plaintext = decrypt(blob, derive_key(passphrase), self.config.capacity)
        try:
            message = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CryptoError("decrypted payload is not valid UTF-8") from exc
        return message, self._metrics(len(token_ids), self.config.payload_bits, entropy)

    def _table(self, session: ModelSession) -> FrequencyTable:
        logits = adjust_logits(
            session.next_logits(),
            session.token_ids,
            self.config.policy,
        )
        ranked = np.lexsort((np.arange(logits.size), -logits))
        probe_count = min(logits.size, max(256, self.config.policy.top_k * 32))
        safe = session.safe_candidates(ranked[:probe_count].tolist())
        return quantize_logits(logits, self.config.policy, safe)

    @staticmethod
    def _entropy(table: FrequencyTable) -> float:
        return -sum(p * math.log2(p) for p in table.probabilities if p > 0)

    @staticmethod
    def _metrics(tokens: int, payload_bits: int, entropy: float) -> CodecMetrics:
        return CodecMetrics(
            tokens=tokens,
            payload_bits=payload_bits,
            bits_per_token=payload_bits / tokens,
            model_entropy_bits=entropy,
            entropy_utilization=payload_bits / entropy if entropy else 0.0,
        )
