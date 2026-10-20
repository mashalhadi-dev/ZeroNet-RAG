"""CLI entry point for offline indexing and query execution."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from src.chunker import chunk_documents
from src.indexer import build_index
from src.parser import walk_corpus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Offline local RAG for AMD AI League Match 3")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", action="store_true", help="parse corpus and build local index")
    mode.add_argument("--query", help="question to answer")
    parser.add_argument("--query-id", help="required with --query; used in output filename")
    parser.add_argument("--corpus", default="/app/corpus")
    parser.add_argument("--index-dir", default="/app/index")
    parser.add_argument("--output-dir", default="/app/output")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.index:
        documents = list(walk_corpus(args.corpus))
        chunks = chunk_documents(documents)
        build_index(chunks, args.index_dir)
        logging.info("Indexed %d documents into %d chunks at %s", len(documents), len(chunks), args.index_dir)
        return 0

    if not args.query_id:
        logging.error("--query-id is required with --query")
        return 2
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", args.query_id):
        logging.error("query-id may contain only letters, numbers, dot, underscore, and hyphen")
        return 2
    from src.agent import RAGAgent
    try:
        result = RAGAgent(args.index_dir).answer(args.query)
    except Exception:
        logging.exception("Query failed; refusing safely")
        result = {"answer": "", "citations": []}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.query_id}_output.json"
    tmp = output_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)
    logging.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
