# LEC — Linguistic Entropy Coding

LEC is an experimental, reversible text-steganography system. It encrypts a
short UTF-8 message with ChaCha20-Poly1305, interprets the ciphertext as entropy,
and uses it to choose tokens from a language model's next-token distribution.
Replaying the same model and arithmetic intervals recovers the exact ciphertext
and authenticated plaintext.

> **Research prototype:** this is not a production secrecy or key-management
> system. Exact text, tokenizer, model weights, runtime configuration, prompt,
> and coding parameters are required for decoding.

## What v0.1 fixes

- model: [`distilbert/distilgpt2`](https://huggingface.co/distilbert/distilgpt2)
  at commit `2290a62682d06624634c1f46a6ad5be0f47f38aa`
- CPU, float32, deterministic PyTorch inference
- fixed generation prompt
- temperature `1.0`, top-k `50`, top-p `0.95`
- repetition penalty `1.25` and a 6-token no-repeat constraint
- 16-bit integer probability tables
- arbitrary-precision arithmetic interval coding
- ChaCha20-Poly1305 with fixed-size plaintext blocks
- tokenization-roundtrip-safe candidate filtering

The included toy model is for fast offline tests. Its output is not intended to
look natural.

## Install

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[model,web,dev]"
```

The first real-model operation downloads approximately 350 MB of model files
from Hugging Face. LEC pins the weight revision, but decoding should still use
the same Python environment and hardware path as encoding.

## Hosted demo

The Vercel deployment uses the deterministic lightweight model because
serverless functions cannot practically load and run the pinned DistilGPT-2
model within their bundle and execution limits. This is labeled in the UI.
Clone the repository and run it locally for the real-model experiment.

## Browser interface

```bash
lec serve
```

Open <http://127.0.0.1:8000>. The server binds only to localhost by default.
Keys and messages are processed in the local Python process and are not stored
by LEC. Select **Offline toy model** to try the complete flow without loading
DistilGPT-2. Covertext appears token by token while encoding. The demo's
advanced controls expose temperature, top-k, top-p, probability precision,
repetition penalty, and the no-repeat phrase length; decoding must use the same values.

## CLI

```bash
lec encode --message "Hello world" --key "demo passphrase" --output cover.txt
lec decode --input cover.txt --key "demo passphrase"
```

For a fast offline round trip:

```bash
lec encode --toy --capacity 16 --message "Hello world" --key secret > cover.txt
lec decode --toy --capacity 16 --input cover.txt --key secret
```

Metrics are written to stderr so stdout can be redirected safely. Capacity is
the maximum UTF-8 plaintext size, not character count, and must match during
decoding.

## How it works

1. Prefix the message with its byte length and pad it to a fixed capacity.
2. Encrypt and authenticate the frame with ChaCha20-Poly1305 and a random nonce.
3. Treat the fixed encrypted block as an integer point in `[0, 2^N)`.
4. At every generation step, deterministically filter and quantize model
   probabilities to positive integer frequencies summing to `2^16`.
5. Select the token whose arithmetic subinterval contains the encrypted point.
6. Stop when the interval contains exactly one integer.
7. To decode, retokenize the unmodified text and replay the same intervals.

Candidates are filtered so the rendered token sequence retokenizes identically.
This is necessary because a tokenizer can otherwise merge adjacent generated
tokens when plain text is read back.

## Reproducibility contract

Successful decoding requires all of the following to match:

- covertext bytes, with no edits, rewrapping, or normalization;
- model repository and exact weight revision;
- tokenizer and Transformers implementation;
- CPU deterministic inference path;
- prompt, temperature, candidate policy, quantization, and payload capacity.

Authenticated decryption turns mismatches into a hard failure rather than
returning plausible but incorrect plaintext.

## Tests

```bash
pytest
```

The offline suite checks Unicode and empty-message round trips, fixed vectors,
integer-frequency invariants, nonce variation, wrong keys, modifications, and
configuration mismatch.

## Experiments

```bash
python experiments/reproducibility.py
python experiments/capacity.py --toy --samples 20 --output results.jsonl
```

The JSONL output records token count, bits/token, entropy utilization, and the
configuration. It is intended as the base data for capacity/quality plots.

## Security and research limitations

- The passphrase KDF uses scrypt with a fixed application salt for v0.1
  reproducibility. Use a random stored salt and proper key management in any
  serious application.
- LEC does not resist paraphrasing, formatting changes, token deletion, Unicode
  normalization, or model/runtime drift.
- It provides no error correction, synchronization, traffic analysis defense,
  or claim of undetectability.
- The fixed payload capacity leaks the chosen configuration; generated token
  count varies.
- Natural-looking model output is not necessarily human-like or statistically
  undetectable.
- This implementation demonstrates the mapping; it does not claim that neural
  linguistic steganography or arithmetic coding is new.

## Paper framing

The draft in `paper/paper.tex` is titled **Linguistic Entropy Coding:
Reversible Ciphertext Embedding Through Language-Model Sampling**. The intended
contributions are deterministic integer coding, measured entropy utilization,
the capacity/naturalness tradeoff, and detectability relative to ordinary
sampling—not a new cryptographic primitive.

## License

MIT. The DistilGPT-2 weights are separately licensed under Apache-2.0.
