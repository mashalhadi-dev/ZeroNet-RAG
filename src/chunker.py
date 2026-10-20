"""Structure-aware paragraph chunking with stable relative-path citations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from .parser import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def chunk_document(document: ParsedDocument, max_chars: int = 1200, overlap_chars: int = 160) -> list[Chunk]:
    if max_chars < 100 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("chunk size must be >= 100 and overlap must be smaller than chunk size")
    # Preserve paragraph/heading boundaries where possible, then split oversized blocks.
    blocks = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", document.text)]
    blocks = [b for b in blocks if b]
    units: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            units.append(block)
        else:
            start = 0
            while start < len(block):
                end = min(start + max_chars, len(block))
                if end < len(block):
                    boundary = block.rfind(" ", start + max_chars // 2, end)
                    if boundary > start:
                        end = boundary
                units.append(block[start:end].strip())
                if end >= len(block):
                    break
                start = max(start + 1, end - overlap_chars)
    result: list[Chunk] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}\n{unit}" if buffer else unit
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                result.append(Chunk(f"{document.path}#{len(result)+1}", document.path, buffer))
            buffer = unit
    if buffer:
        result.append(Chunk(f"{document.path}#{len(result)+1}", document.path, buffer))
    return result


def chunk_documents(documents: list[ParsedDocument], **kwargs) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, **kwargs)]
