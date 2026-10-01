"""Deterministic language-model adapters used by both encoder and decoder."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class ModelError(Exception):
    pass


class ModelSession(ABC):
    @property
    @abstractmethod
    def token_ids(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def next_logits(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def append(self, token_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def safe_candidates(self, ranked_ids: Sequence[int]) -> set[int]:
        raise NotImplementedError


class LanguageModel(ABC):
    @abstractmethod
    def start(self, prompt: str) -> ModelSession:
        raise NotImplementedError

    @abstractmethod
    def render(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse(self, text: str) -> list[int]:
        raise NotImplementedError


class TransformersModel(LanguageModel):
    """Pinned Hugging Face causal LM running deterministically on CPU."""

    def __init__(self, model_id: str, revision: str = "main") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelError(
                "model dependencies are missing; install with `pip install -e '.[model]'`"
            ) from exc

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.clean_up_tokenization_spaces = False
        self.network = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float32,
        ).to("cpu")
        self.network.eval()

    def start(self, prompt: str) -> ModelSession:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise ModelError("prompt tokenized to an empty sequence")
        return _TransformersSession(self, prompt_ids)

    def render(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def parse(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class _TransformersSession(ModelSession):
    def __init__(self, owner: TransformersModel, prompt_ids: list[int]) -> None:
        self.owner = owner
        self._token_ids: list[int] = []
        tensor = owner._torch.tensor([prompt_ids], dtype=owner._torch.long)
        with owner._torch.inference_mode():
            output = owner.network(input_ids=tensor, use_cache=True)
        self._past = output.past_key_values
        self._logits = output.logits[0, -1].float().cpu().numpy()

    @property
    def token_ids(self) -> list[int]:
        return self._token_ids

    def next_logits(self) -> np.ndarray:
        return self._logits

    def append(self, token_id: int) -> None:
        tensor = self.owner._torch.tensor([[token_id]], dtype=self.owner._torch.long)
        with self.owner._torch.inference_mode():
            output = self.owner.network(
                input_ids=tensor,
                past_key_values=self._past,
                use_cache=True,
            )
        self._past = output.past_key_values
        self._logits = output.logits[0, -1].float().cpu().numpy()
        self._token_ids.append(token_id)

    def safe_candidates(self, ranked_ids: Sequence[int]) -> set[int]:
        safe: set[int] = set()
        for token_id in ranked_ids:
            candidate = self._token_ids + [int(token_id)]
            text = self.owner.render(candidate)
            if self.owner.parse(text) == candidate:
                safe.add(int(token_id))
        return safe


class ToyLanguageModel(LanguageModel):
    """Small deterministic model for tests and an offline smoke demo."""

    words = (
        "the", "quiet", "train", "moved", "through", "a", "green", "valley",
        "while", "evening", "light", "settled", "over", "distant", "hills",
        "and", "people", "watched", "from", "their", "windows", "as", "soft",
        "rain", "began", "to", "fall", "near", "old", "stone", "houses", "today",
    )

    def start(self, prompt: str) -> ModelSession:
        return _ToySession(self, prompt)

    def render(self, token_ids: Sequence[int]) -> str:
        return " ".join(self.words[token] for token in token_ids)

    def parse(self, text: str) -> list[int]:
        lookup = {word: index for index, word in enumerate(self.words)}
        try:
            return [lookup[word] for word in text.split()]
        except KeyError as exc:
            raise ModelError(f"unknown toy-model token: {exc.args[0]}") from exc


class _ToySession(ModelSession):
    def __init__(self, owner: ToyLanguageModel, prompt: str) -> None:
        self.owner = owner
        self.prompt = prompt
        self._token_ids: list[int] = []

    @property
    def token_ids(self) -> list[int]:
        return self._token_ids

    def next_logits(self) -> np.ndarray:
        seed_material = self.prompt.encode() + bytes(self._token_ids)
        digest = hashlib.sha256(seed_material).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)
        return (values[: len(self.owner.words)] - 128.0) / 24.0

    def append(self, token_id: int) -> None:
        self._token_ids.append(token_id)

    def safe_candidates(self, ranked_ids: Sequence[int]) -> set[int]:
        return {int(token) for token in ranked_ids}
