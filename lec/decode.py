"""Public decoding convenience function."""

from .codec import CodecMetrics, LECCodec
from .config import LECConfig


def decode(
    text: str,
    key: str,
    config: LECConfig | None = None,
) -> tuple[str, CodecMetrics]:
    return LECCodec(config=config).decode(text, key)
