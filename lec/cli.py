"""Command-line interface for Linguistic Entropy Coding."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Sequence

from .codec import LECCodec
from .config import LECConfig
from .model import ToyLanguageModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lec",
        description="Reversibly embed encrypted text in language-model output.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encode = commands.add_parser("encode", help="encode a message as covertext")
    source = encode.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="message to encode")
    source.add_argument("--input", type=Path, help="read the message from a UTF-8 file")
    _common_codec_options(encode)
    encode.add_argument("--output", type=Path, help="write covertext to this file")

    decode = commands.add_parser("decode", help="decode covertext")
    source = decode.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="covertext to decode")
    source.add_argument("--input", type=Path, help="read covertext from a UTF-8 file")
    _common_codec_options(decode)

    serve = commands.add_parser("serve", help="run the local web interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def _common_codec_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key",
        help="encryption key (omit to enter it without terminal echo)",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=LECConfig().capacity,
        help="maximum UTF-8 message bytes (default: %(default)s)",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="use the deterministic offline toy model",
    )


def _read_text(value: str | None, path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    assert value is not None
    return value


def _codec(capacity: int, toy: bool) -> LECCodec:
    config = LECConfig(capacity=capacity)
    model = ToyLanguageModel() if toy else None
    return LECCodec(config=config, model=model)


def _key(value: str | None) -> str:
    key = value if value is not None else getpass.getpass("Key: ")
    if not key:
        raise ValueError("key must not be empty")
    return key


def _print_metrics(metrics: object) -> None:
    to_dict = getattr(metrics, "to_dict")
    print(json.dumps(to_dict(), indent=2), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "serve":
            try:
                import uvicorn
            except ImportError as exc:
                raise RuntimeError(
                    "web dependencies are missing; install with `pip install -e '.[web]'`"
                ) from exc
            uvicorn.run("lec.web:app", host=args.host, port=args.port, reload=args.reload)
            return 0

        key = _key(args.key)
        codec = _codec(args.capacity, args.toy)
        if args.command == "encode":
            result = codec.encode(_read_text(args.message, args.input), key)
            if args.output is not None:
                args.output.write_text(result.text, encoding="utf-8")
            else:
                sys.stdout.write(result.text)
            _print_metrics(result.metrics)
        else:
            message, metrics = codec.decode(
                _read_text(args.text, args.input),
                key,
            )
            sys.stdout.write(message)
            _print_metrics(metrics)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"lec: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Core errors are safe to show; no message or key is interpolated here.
        print(f"lec: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
