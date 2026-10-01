"""The deliberately narrow, shared v0.1 configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .probabilities import SamplingPolicy

MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
PROMPT = (
    "The afternoon was ordinary at first. People moved through the neighborhood, "
    "and the details of the day unfolded one after another. "
)


@dataclass(frozen=True)
class LECConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    prompt: str = PROMPT
    capacity: int = 16
    max_tokens: int = 1024
    policy: SamplingPolicy = field(default_factory=SamplingPolicy)

    @property
    def payload_bytes(self) -> int:
        # nonce + four-byte length prefix + padded message + authentication tag
        return 12 + 4 + self.capacity + 16

    @property
    def payload_bits(self) -> int:
        return self.payload_bytes * 8

    def validate(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.policy.validate()
