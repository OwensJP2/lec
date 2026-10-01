import pytest

from lec.crypto import CryptoError, blob_size, decrypt, derive_key, encrypt


def test_fixed_block_layout_and_roundtrip() -> None:
    key = derive_key("test key")
    blob = encrypt(b"hello", key, capacity=8, nonce=bytes(12))
    assert len(blob.data) == blob_size(8)
    assert decrypt(blob.data, key, 8) == b"hello"


def test_empty_key_rejected() -> None:
    with pytest.raises(CryptoError):
        derive_key("")
