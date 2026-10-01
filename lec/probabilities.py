"""Deterministic candidate selection and integer probability quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    precision_bits: int = 16
    repetition_penalty: float = 1.25
    no_repeat_ngram: int = 6

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 2:
            raise ValueError("top_k must be at least 2")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 8 <= self.precision_bits <= 24:
            raise ValueError("precision_bits must be between 8 and 24")
        if not 1 <= self.repetition_penalty <= 2:
            raise ValueError("repetition_penalty must be between 1 and 2")
        if not 0 <= self.no_repeat_ngram <= 16:
            raise ValueError("no_repeat_ngram must be between 0 and 16")


@dataclass(frozen=True)
class FrequencyTable:
    token_ids: tuple[int, ...]
    frequencies: tuple[int, ...]
    cumulative: tuple[int, ...]
    total: int
    probabilities: tuple[float, ...]

    def interval(self, token_id: int) -> tuple[int, int]:
        try:
            index = self.token_ids.index(token_id)
        except ValueError as exc:
            raise ValueError(f"token {token_id} is outside the candidate set") from exc
        return self.cumulative[index], self.cumulative[index + 1]


def adjust_logits(
    logits: np.ndarray,
    token_ids: list[int],
    policy: SamplingPolicy,
) -> np.ndarray:
    """Apply history-dependent controls before quantization.

    Both operations are deterministic functions of the shared token history, so
    the decoder reconstructs the same candidate intervals.
    """
    scores = np.array(logits, dtype=np.float64, copy=True)
    if policy.repetition_penalty > 1 and token_ids:
        for token in set(token_ids):
            if scores[token] < 0:
                scores[token] *= policy.repetition_penalty
            else:
                scores[token] /= policy.repetition_penalty
    n = policy.no_repeat_ngram
    if n >= 2 and len(token_ids) >= n - 1:
        prefix = tuple(token_ids[-(n - 1) :])
        for start in range(len(token_ids) - n + 1):
            if tuple(token_ids[start : start + n - 1]) == prefix:
                scores[token_ids[start + n - 1]] = -1e9
    return scores


def quantize_logits(
    logits: np.ndarray,
    policy: SamplingPolicy,
    allowed_token_ids: set[int] | None = None,
) -> FrequencyTable:
    """Convert logits into a stable token-id-ordered integer table.

    Selection uses descending score with token id as the tie-breaker. Largest
    remainder quantization then produces positive frequencies summing exactly
    to ``2**precision_bits``.
    """
    policy.validate()
    scores = np.asarray(logits, dtype=np.float64).reshape(-1) / policy.temperature
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must be finite")

    ranked = np.lexsort((np.arange(scores.size), -scores)).tolist()
    if allowed_token_ids is not None:
        ranked = [token for token in ranked if token in allowed_token_ids]
    ranked = ranked[: policy.top_k]
    if len(ranked) < 2:
        raise ValueError("candidate policy produced fewer than two tokens")

    selected_scores = np.asarray([scores[token] for token in ranked])
    selected_scores -= selected_scores.max()
    probs = np.exp(selected_scores)
    probs /= probs.sum()

    if policy.top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), policy.top_p, side="left")) + 1
        keep = max(2, keep)
        ranked = ranked[:keep]
        probs = probs[:keep]
        probs /= probs.sum()

    total = 1 << policy.precision_bits
    if len(ranked) > total:
        raise ValueError("frequency precision is too small for candidate set")
    remaining = total - len(ranked)
    quotas = probs * remaining
    extras = np.floor(quotas).astype(np.int64)
    frequencies = extras + 1
    shortfall = total - int(frequencies.sum())
    remainders = quotas - extras
    award_order = sorted(
        range(len(ranked)),
        key=lambda index: (-remainders[index], ranked[index]),
    )
    for index in award_order[:shortfall]:
        frequencies[index] += 1

    # Arithmetic intervals are ordered by token id, independent of ranking.
    ordered = sorted(
        zip(ranked, frequencies.tolist(), probs.tolist()),
        key=lambda item: item[0],
    )
    token_ids = tuple(item[0] for item in ordered)
    freqs = tuple(int(item[1]) for item in ordered)
    normalized = tuple(float(item[2]) for item in ordered)
    cumulative = [0]
    for frequency in freqs:
        cumulative.append(cumulative[-1] + frequency)
    if cumulative[-1] != total:
        raise AssertionError("quantized frequencies do not sum to target")
    return FrequencyTable(
        token_ids=token_ids,
        frequencies=freqs,
        cumulative=tuple(cumulative),
        total=total,
        probabilities=normalized,
    )