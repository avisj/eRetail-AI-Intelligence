"""Deterministic Document Chunking Engine (Phase 7F).

Splits business knowledge documents into stable, version-linked, heading-aware
chunks without relying on external models or non-deterministic heuristics.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument, KnowledgeChunk


_HEADER_PATTERN = re.compile(r"^(#{1,6}\s+.*|[A-Z0-9\.\s]{3,}:)$")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class DeterministicChunker:
    """Configurable deterministic chunker for business documents."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        if chunk_size <= 50:
            raise ValueError("chunk_size must be greater than 50 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: BusinessKnowledgeDocument) -> List[KnowledgeChunk]:
        """Split a document into deterministic KnowledgeChunk instances."""
        raw_text = document.content.strip()
        if not raw_text:
            return []

        lines = raw_text.splitlines()
        segments: List[Tuple[Optional[str], str]] = []

        current_heading: Optional[str] = None
        current_block: List[str] = []

        for line in lines:
            line_str = line.strip()
            if _HEADER_PATTERN.match(line_str):
                # Flush previous block
                if current_block:
                    block_text = "\n".join(current_block).strip()
                    if block_text:
                        segments.append((current_heading, block_text))
                    current_block = []
                # Update current heading
                clean_header = line_str.lstrip("#").strip()
                current_heading = clean_header
            elif not line_str:
                # Empty line -> paragraph break
                if current_block:
                    block_text = "\n".join(current_block).strip()
                    if block_text:
                        segments.append((current_heading, block_text))
                    current_block = []
            else:
                current_block.append(line_str)

        if current_block:
            block_text = "\n".join(current_block).strip()
            if block_text:
                segments.append((current_heading, block_text))

        # Now assemble segments into chunks respecting chunk_size and chunk_overlap
        chunks: List[KnowledgeChunk] = []
        seq_num = 0

        buffer_text = ""
        buffer_heading: Optional[str] = None

        for heading, text in segments:
            # If the segment text itself exceeds chunk_size, split it along sentence boundaries
            if len(text) > self.chunk_size:
                if buffer_text.strip():
                    chunk = self._create_chunk(
                        document=document,
                        seq_num=seq_num,
                        content=buffer_text.strip(),
                        heading=buffer_heading,
                    )
                    chunks.append(chunk)
                    seq_num += 1
                    buffer_text = ""
                    buffer_heading = None

                sub_chunks = self._split_long_text(text, self.chunk_size, self.chunk_overlap)
                for idx, sub_txt in enumerate(sub_chunks):
                    if idx == len(sub_chunks) - 1:
                        buffer_text = sub_txt
                        buffer_heading = heading
                    else:
                        chunk = self._create_chunk(
                            document=document,
                            seq_num=seq_num,
                            content=sub_txt.strip(),
                            heading=heading,
                        )
                        chunks.append(chunk)
                        seq_num += 1
                continue

            if not buffer_text:
                buffer_text = text
                buffer_heading = heading
            elif heading and buffer_heading and heading != buffer_heading:
                # Flush buffer on distinct section heading boundary
                chunk = self._create_chunk(
                    document=document,
                    seq_num=seq_num,
                    content=buffer_text.strip(),
                    heading=buffer_heading,
                )
                chunks.append(chunk)
                seq_num += 1
                buffer_text = text
                buffer_heading = heading
            elif len(buffer_text) + len(text) + 2 <= self.chunk_size:
                buffer_text += "\n\n" + text
                if not buffer_heading and heading:
                    buffer_heading = heading
            else:
                chunk = self._create_chunk(
                    document=document,
                    seq_num=seq_num,
                    content=buffer_text.strip(),
                    heading=buffer_heading,
                )
                chunks.append(chunk)
                seq_num += 1
                buffer_text = text
                buffer_heading = heading

        if buffer_text.strip():
            chunk = self._create_chunk(
                document=document,
                seq_num=seq_num,
                content=buffer_text.strip(),
                heading=buffer_heading,
            )
            chunks.append(chunk)

        return chunks

    def _split_long_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split an oversized block into smaller pieces along sentence boundaries."""
        sentences = _SENTENCE_SPLIT_PATTERN.split(text)
        sub_chunks: List[str] = []
        cur_sub = ""

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if not cur_sub:
                cur_sub = s_clean
            elif len(cur_sub) + len(s_clean) + 1 <= max_size:
                cur_sub += " " + s_clean
            else:
                sub_chunks.append(cur_sub)
                if overlap > 0 and len(cur_sub) > overlap:
                    prefix = cur_sub[-overlap:]
                    space_idx = prefix.find(" ")
                    if space_idx != -1:
                        prefix = prefix[space_idx + 1:]
                    cur_sub = prefix + " " + s_clean
                else:
                    cur_sub = s_clean

        if cur_sub:
            sub_chunks.append(cur_sub)
        return sub_chunks

    def _create_chunk(
        self,
        document: BusinessKnowledgeDocument,
        seq_num: int,
        content: str,
        heading: Optional[str],
    ) -> KnowledgeChunk:
        """Construct a stable, deterministic KnowledgeChunk."""
        clean_content = content.strip()
        chk_id = f"CHK-{document.document_id}-v{document.version}-{seq_num:04d}"
        chk_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()

        return KnowledgeChunk(
            chunk_id=chk_id,
            document_id=document.document_id,
            document_version=document.version,
            sequence_number=seq_num,
            title=document.title,
            section_heading=heading,
            page_or_location=f"Section: {heading}" if heading else f"Chunk #{seq_num}",
            content=clean_content,
            checksum=chk_hash,
            organization_id=document.organization_id,
            domain=document.domain,
            document_type=document.document_type,
            effective_from=document.effective_from,
            effective_to=document.effective_to,
            status=document.status,
            tags=list(document.tags),
        )
