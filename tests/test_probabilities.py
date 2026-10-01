import numpy as np

from lec.probabilities import SamplingPolicy, adjust_logits, quantize_logits


def test_quantization_is_exact_and_deterministic() -> None:
    logits = np.array([0.1, 2.0, 1.0, -3.0, 2.0])
    policy = SamplingPolicy(top_k=4, precision_bits=10)
    first = quantize_logits(logits, policy)
    second = quantize_logits(logits, policy)
    assert first == second
    assert sum(first.frequencies) == 2**10
    assert all(frequency > 0 for frequency in first.frequencies)
    assert first.token_ids == tuple(sorted(first.token_ids))


def test_allowed_candidate_filter() -> None:
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    table = quantize_logits(
        logits,
        SamplingPolicy(top_k=2, precision_bits=8),
        allowed_token_ids={1, 2, 3},
    )
    assert set(table.token_ids) == {1, 2}


def test_repetition_controls_downrank_and_ban_phrases() -> None:
    logits = np.linspace(-1, 2, 6)
    history = [0, 1, 2, 0, 1]
    policy = SamplingPolicy(
        top_k=4,
        precision_bits=10,
        repetition_penalty=1.5,
        no_repeat_ngram=3,
    )
    adjusted = adjust_logits(logits, history, policy)
    assert adjusted[2] < -1e8
    assert adjusted[0] < logits[0]
    table = quantize_logits(adjusted, policy)
    assert 2 not in table.token_ids
