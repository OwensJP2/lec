"""Exact arbitrary-precision arithmetic interval coding.

The encrypted block is interpreted as one point in a ``2**N`` interval.
Successive language-model frequency tables narrow that interval until exactly
one point remains. Replaying the same tables and tokens reconstructs the point.
"""

from __future__ import annotations

from dataclasses import dataclass

from .probabilities import FrequencyTable


class CodingError(Exception):
    """The token stream cannot represent the configured payload."""


@dataclass
class Interval:
    low: int
    high: int

    @classmethod
    def for_bits(cls, bit_count: int) -> "Interval":
        if bit_count < 1:
            raise ValueError("bit_count must be positive")
        return cls(0, 1 << bit_count)

    @property
    def width(self) -> int:
        return self.high - self.low

    @property
    def complete(self) -> bool:
        return self.width == 1

    def choose(self, value: int, table: FrequencyTable) -> int:
        if not self.low <= value < self.high:
            raise CodingError("payload point is outside the current interval")
        for index, token_id in enumerate(table.token_ids):
            child_low, child_high = self._child(index, table)
            if child_low <= value < child_high:
                self.low, self.high = child_low, child_high
                return token_id
        raise CodingError("integer quantization left the payload point uncovered")

    def consume(self, token_id: int, table: FrequencyTable) -> None:
        try:
            index = table.token_ids.index(token_id)
        except ValueError as exc:
            raise CodingError(
                "covertext token is not valid for the reconstructed distribution"
            ) from exc
        child_low, child_high = self._child(index, table)
        if child_low >= child_high:
            raise CodingError("covertext selected an empty arithmetic interval")
        self.low, self.high = child_low, child_high

    def _child(self, index: int, table: FrequencyTable) -> tuple[int, int]:
        width = self.width
        start = self.low + width * table.cumulative[index] // table.total
        end = self.low + width * table.cumulative[index + 1] // table.total
        return start, end


def bytes_to_point(data: bytes) -> int:
    return int.from_bytes(data, "big")


def point_to_bytes(point: int, byte_count: int) -> bytes:
    if not 0 <= point < 1 << (byte_count * 8):
        raise CodingError("decoded point does not fit the payload block")
    return point.to_bytes(byte_count, "big")
