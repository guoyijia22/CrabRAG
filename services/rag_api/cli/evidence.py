from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from services.rag_api.retrieval.evidence_service import retrieve_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except CliArgumentError as exc:
        _write_json({"ok": False, "error": str(exc)}, stream=sys.stderr, pretty=False)
        return 2
    try:
        payload = retrieve_evidence(
            question=args.question,
            top_k=args.top_k,
            mode=args.mode,
            include_trace=args.include_trace,
            no_rerank=args.no_rerank,
        )
    except Exception as exc:  # noqa: BLE001
        _write_json({"ok": False, "error": str(exc)}, stream=sys.stderr, pretty=args.pretty)
        return 1
    _write_json(payload, stream=sys.stdout, pretty=args.pretty)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description="Query local RAG evidence as JSON.")
    parser.add_argument("--question", required=True, help="用户问题")
    parser.add_argument("--top-k", type=int, default=None, help="返回证据片段数量，范围 1-10")
    parser.add_argument("--mode", choices=["auto", "vector", "graph", "hybrid"], default="auto", help="检索模式")
    parser.add_argument("--pretty", action="store_true", help="格式化 JSON 输出")
    parser.add_argument("--include-trace", action="store_true", help="包含检索 trace")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Rerank 重排")
    return parser


class CliArgumentError(ValueError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _write_json(payload: dict, *, stream, pretty: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
    stream.write(text)
    stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
