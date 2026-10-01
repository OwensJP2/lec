"""Public encoding convenience function."""

from .codec import EncodeResult, LECCodec
from .config import LECConfig


def encode(
    message: str,
    key: str,
    config: LECConfig | None = None,
) -> EncodeResult:
    return LECCodec(config=config).encode(message, key)
