"""Authenticated encryption and fixed-size payload framing for LEC.

LEC does not introduce a new cryptographic primitive. Encryption is handled by
a standard authenticated cipher (ChaCha20-Poly1305). This module also defines
the *framing*: how a UTF-8 plaintext is turned into a fixed-size byte blob that
the entropy coder can embed, and back again.

Blob layout (all fixed length for a given ``capacity``)::

    nonce (12 bytes) | ciphertext+tag (capacity + 16 bytes)

where the encrypted plaintext frame is::

    length (4 bytes, big-endian) | utf-8 plaintext | zero padding -> capacity

The 4-byte length prefix lets the decoder strip the zero padding exactly, so
recovery is byte-exact. ``capacity`` is a shared configuration parameter: the
encoder and decoder must agree on it (it is carried in :class:`lec.config.LECConfig`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

NONCE_BYTES = 12
TAG_BYTES = 16
LENGTH_PREFIX_BYTES = 4

# Fixed salt for the demonstration key-derivation function. v0.1 deliberately
# avoids production key management; a passphrase is stretched with scrypt using
# a fixed salt so the same passphrase always yields the same key. This is fine
# for reproducible experiments but is NOT a substitute for real key management.
_KDF_SALT = b"lec-v0.1-fixed-salt"
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1


class CryptoError(Exception):
    """Raised on authentication failure or malformed / oversized input."""


def derive_key(passphrase: str) -> bytes:
    """Derive a 32-byte key from a passphrase using scrypt with a fixed salt."""
    if not passphrase:
        raise CryptoError("key must not be empty")
    kdf = Scrypt(salt=_KDF_SALT, length=32, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


@dataclass(frozen=True)
class Blob:
    """A fixed-size encrypted payload ready for entropy coding."""

    data: bytes

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.data)


def blob_size(capacity: int) -> int:
    """Total fixed blob size in bytes for a given plaintext ``capacity``."""
    return NONCE_BYTES + LENGTH_PREFIX_BYTES + capacity + TAG_BYTES


def frame_plaintext(plaintext: bytes, capacity: int) -> bytes:
    """Prefix with a 4-byte length and zero-pad to ``capacity`` bytes."""
    if capacity < 1:
        raise CryptoError("capacity must be positive")
    if len(plaintext) > capacity:
        raise CryptoError(
            f"message is {len(plaintext)} bytes but capacity is {capacity}; "
            f"increase --capacity"
        )
    header = len(plaintext).to_bytes(LENGTH_PREFIX_BYTES, "big")
    body = plaintext + b"\x00" * (capacity - len(plaintext))
    return header + body


def unframe_plaintext(frame: bytes, capacity: int) -> bytes:
    """Inverse of :func:`frame_plaintext`."""
    if len(frame) != LENGTH_PREFIX_BYTES + capacity:
        raise CryptoError("framed plaintext has unexpected length")
    n = int.from_bytes(frame[:LENGTH_PREFIX_BYTES], "big")
    if n > capacity:
        raise CryptoError("corrupt length prefix")
    return frame[LENGTH_PREFIX_BYTES : LENGTH_PREFIX_BYTES + n]


def encrypt(plaintext: bytes, key: bytes, capacity: int, nonce: bytes | None = None) -> Blob:
    """Encrypt ``plaintext`` into a fixed-size :class:`Blob`.

    A random nonce is generated unless one is supplied (tests / vectors pass a
    fixed nonce for determinism).
    """
    if nonce is None:
        nonce = os.urandom(NONCE_BYTES)
    if len(nonce) != NONCE_BYTES:
        raise CryptoError("nonce must be 12 bytes")
    frame = frame_plaintext(plaintext, capacity)
    ct = ChaCha20Poly1305(key).encrypt(nonce, frame, None)
    return Blob(nonce + ct)


def decrypt(blob: bytes, key: bytes, capacity: int) -> bytes:
    """Decrypt and authenticate a fixed-size blob back to the plaintext."""
    expected = blob_size(capacity)
    if len(blob) != expected:
        raise CryptoError(f"blob is {len(blob)} bytes, expected {expected}")
    nonce, ct = blob[:NONCE_BYTES], blob[NONCE_BYTES:]
    try:
        frame = ChaCha20Poly1305(key).decrypt(nonce, ct, None)
    except Exception as exc:  # cryptography raises InvalidTag
        raise CryptoError("authentication failed (wrong key or corrupt text)") from exc
    return unframe_plaintext(frame, capacity)
