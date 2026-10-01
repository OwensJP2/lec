import hashlib

from lec.codec import LECCodec
from lec.config import LECConfig
from lec.model import ToyLanguageModel
from lec.probabilities import SamplingPolicy


def test_v01_toy_vector() -> None:
    config = LECConfig(
        capacity=4,
        policy=SamplingPolicy(
            top_k=16,
            precision_bits=12,
            repetition_penalty=1.0,
            no_repeat_ngram=0,
        ),
    )
    codec = LECCodec(config, ToyLanguageModel())
    result = codec.encode("LEC", "vector-key", nonce=bytes(range(12)))
    digest = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
    assert digest == "f0a511b6bf025c0b6681ff94fe4ed934532645128c0119f1537914029763b2c2"
    assert result.metrics.tokens == 109
    assert codec.decode(result.text, "vector-key")[0] == "LEC"
