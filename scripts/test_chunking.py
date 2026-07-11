import argparse
from pathlib import Path
import re
from statistics import median
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.vector_store import TokenizerLike, VectorStore, chunk_markdown


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class WordTokenizer:
    def __init__(self) -> None:
        self._pieces: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pieces = re.findall(r"\w+|[^\w\s]", text)
        token_ids: list[int] = []
        for piece in pieces:
            token_id = len(self._pieces) + 1
            self._pieces[token_id] = piece
            token_ids.append(token_id)
        return token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return " ".join(self._pieces[token_id] for token_id in token_ids)


CHAR_TOKENIZER = CharacterTokenizer()


def assert_synthetic_markdown() -> None:
    sample = """# Product DB

## Setup

Use PostgreSQL for transactional metadata and Qdrant for retrieval.

```python
def choose_store(workload):
    if workload == "semantic search":
        return "qdrant"
    return "postgresql"
```

| Store | Use |
| --- | --- |
| PostgreSQL | OLTP |
| Qdrant | Vector search |
"""
    chunks = chunk_markdown(
        sample,
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=120,
        body_max_tokens=120,
        overlap_target_tokens=30,
        overlap_max_tokens=30,
    )
    content_types = {chunk.content_type for chunk in chunks}

    if {"text", "code", "table"} - content_types:
        raise AssertionError(f"Missing expected content types: {content_types}")

    for chunk in chunks:
        if chunk.h1 != "Product DB" or chunk.h2 != "Setup":
            raise AssertionError(f"Heading metadata missing: {chunk}")
        if chunk.content_type == "text" and chunk.text.lstrip().startswith("#"):
            raise AssertionError("Text payload should not duplicate headings.")
        if chunk.content_type == "code":
            stripped = chunk.text.strip()
            if not stripped.startswith("```") or not stripped.endswith("```"):
                raise AssertionError("Code chunks should preserve fenced code blocks.")

    print("Synthetic structured Markdown -> ok")


def assert_overlap_budget() -> None:
    sentences = [
        f"Sentence {index} carries boundary context words."
        for index in range(12)
    ]
    sample = "# Overlap\n\n" + " ".join(sentences)
    chunks = chunk_markdown(
        sample,
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=100,
        body_max_tokens=120,
        overlap_target_tokens=40,
        overlap_max_tokens=50,
    )
    text_chunks = [chunk for chunk in chunks if chunk.content_type == "text"]
    if len(text_chunks) < 3:
        raise AssertionError("Expected long text to split into multiple chunks.")

    oversized = [len(chunk.text) for chunk in text_chunks if len(chunk.text) > 220]
    if oversized:
        raise AssertionError(f"Text chunks exceeded the total token budget: {oversized}")

    if sentences[1] not in text_chunks[1].text:
        raise AssertionError("A chunk should include complete previous-context sentences.")
    if sentences[2] not in text_chunks[0].text:
        raise AssertionError("A chunk should include complete following-context sentences.")
    for chunk in text_chunks:
        if any(not part.endswith(".") for part in chunk.text.split("\n\n")):
            raise AssertionError("Overlap must preserve complete sentence boundaries.")

    print("Bidirectional sentence overlap -> ok")


def assert_heading_boundaries() -> None:
    alpha = " ".join(
        f"ALPHA sentence {index} stays in its own section."
        for index in range(8)
    )
    beta = " ".join(
        f"BETA sentence {index} stays in its own section."
        for index in range(8)
    )
    sample = f"# Guide\n\n## Alpha\n\n{alpha}\n\n## Beta\n\n{beta}"
    chunks = chunk_markdown(
        sample,
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=150,
        body_max_tokens=160,
        overlap_target_tokens=30,
        overlap_max_tokens=40,
    )
    if not chunks:
        raise AssertionError("Heading-boundary sample should produce chunks.")
    if any("ALPHA" in chunk.text and "BETA" in chunk.text for chunk in chunks):
        raise AssertionError("Chunks and overlaps must not cross Markdown headings.")
    if {chunk.h2 for chunk in chunks} != {"Alpha", "Beta"}:
        raise AssertionError("Section heading metadata should survive paragraph packing.")

    print("Heading boundaries -> ok")


def assert_yaml_frontmatter() -> None:
    sample = """---
id: doc_00000
title: "Software License Agreement"
source: RAGBench/cuad
---

The agreement grants a limited software license.

Additional locations require additional fees.
"""
    chunks = chunk_markdown(
        sample,
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=400,
        body_max_tokens=520,
        overlap_target_tokens=30,
        overlap_max_tokens=40,
    )
    if len(chunks) != 1:
        raise AssertionError(f"Frontmatter sample should produce one chunk: {chunks}")
    chunk = chunks[0]
    if chunk.h1 != "Software License Agreement":
        raise AssertionError("Frontmatter title should become document heading metadata.")
    if "doc_00000" in chunk.text or "---" in chunk.text:
        raise AssertionError("YAML frontmatter must not leak into stored chunk text.")
    if chunk.start_line != 7 or chunk.end_line != 9:
        raise AssertionError(
            f"Frontmatter removal should preserve original line numbers: {chunk}"
        )

    print("YAML frontmatter -> ok")


def assert_indented_prose_is_text() -> None:
    sample = """# Contract

    This Agreement remains ordinary prose despite its visual indentation.

    The parties agree to preserve paragraph-aware retrieval behavior.
"""
    chunks = chunk_markdown(
        sample,
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=400,
        body_max_tokens=520,
        overlap_target_tokens=30,
        overlap_max_tokens=40,
    )
    if not chunks or any(chunk.content_type != "text" for chunk in chunks):
        raise AssertionError("Indented long-form prose should not become code chunks.")

    print("Indented prose normalization -> ok")


def assert_long_paragraph_uses_sentence_boundaries() -> None:
    sentences = [
        f"Long paragraph sentence {index} contains enough words for splitting."
        for index in range(14)
    ]
    chunks = chunk_markdown(
        "# Long\n\n" + " ".join(sentences),
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=180,
        body_max_tokens=240,
        overlap_target_tokens=0,
        overlap_max_tokens=0,
    )
    if len(chunks) < 2:
        raise AssertionError("The oversized paragraph should split into multiple chunks.")
    if any(len(chunk.text) > 240 for chunk in chunks):
        raise AssertionError("Sentence-split chunks must respect the maximum size.")
    for sentence in sentences:
        if sum(sentence in chunk.text for chunk in chunks) != 1:
            raise AssertionError(f"Sentence was cut or duplicated: {sentence}")
    if any(not chunk.text.endswith(".") for chunk in chunks):
        raise AssertionError("Long paragraphs must split on sentence boundaries.")

    print("Long paragraph sentence splitting -> ok")


def assert_metadata_filter() -> None:
    query_filter = VectorStore._metadata_filter(
        {"h2": "Setup", "content_type": "code"}
    )
    if query_filter is None or len(query_filter.must or []) != 2:
        raise AssertionError("Metadata filter should include requested conditions.")

    print("Metadata filter -> ok")


def assert_gfm_like_markdown() -> None:
    sample = """# GFM

| Feature | Status |
| --- | --- |
| Tables | ~~legacy~~ current |

- [x] Keep task list text
- [ ] Keep unchecked task
"""
    chunks = chunk_markdown(
        sample,
        tokenizer=CHAR_TOKENIZER,
        body_target_tokens=200,
        body_max_tokens=200,
        overlap_target_tokens=20,
        overlap_max_tokens=20,
    )
    if not any(chunk.content_type == "table" for chunk in chunks):
        raise AssertionError("GFM-style tables should be parsed as table blocks.")

    combined_text = "\n".join(chunk.text for chunk in chunks)
    if "~~legacy~~" not in combined_text:
        raise AssertionError("Strikethrough text should remain in chunk payload.")
    if "[x] Keep task list text" not in combined_text:
        raise AssertionError("Task-list text should remain retrievable.")

    print("GFM-like Markdown -> ok")


def assert_token_aware_budget() -> None:
    tokenizer = WordTokenizer()
    long_words = " ".join(f"extraordinarilylongword{index}" for index in range(30))
    chunks = chunk_markdown(
        f"# Tokens\n\n{long_words}.",
        tokenizer=tokenizer,
        body_target_tokens=16,
        body_max_tokens=16,
        overlap_target_tokens=0,
        overlap_max_tokens=0,
    )
    if len(chunks) < 2:
        raise AssertionError("Word-token budget should split the oversized paragraph.")
    if any(len(tokenizer.encode(chunk.text)) > 16 for chunk in chunks):
        raise AssertionError("Chunk bodies must respect the configured token maximum.")
    if any(chunk.body_token_count > 16 or chunk.token_count > 16 for chunk in chunks):
        raise AssertionError("Stored token-count metadata must respect body limits.")
    if not any(len(chunk.text) > 16 for chunk in chunks):
        raise AssertionError("Token-aware chunks should not be limited by character count.")

    print("Token-aware body budget -> ok")


def assert_corpus(tokenizer: TokenizerLike, *, tokenizer_label: str) -> None:
    total_chunks = 0
    body_token_counts: list[int] = []
    total_token_counts: list[int] = []

    for path in sorted(settings.docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            tokenizer=tokenizer,
            body_target_tokens=settings.chunk_body_target_tokens,
            body_max_tokens=settings.chunk_body_max_tokens,
            overlap_target_tokens=settings.chunk_overlap_target_tokens,
            overlap_max_tokens=settings.chunk_overlap_max_tokens,
        )

        if not chunks:
            raise AssertionError(f"No chunks produced for {path}")

        if text.lstrip().startswith("#") and not any(chunk.h1 for chunk in chunks):
            raise AssertionError(f"Heading metadata missing in {path}")

        oversized = [
            chunk.token_count
            for chunk in chunks
            if chunk.token_count > settings.chunk_total_max_tokens
        ]
        if oversized:
            raise AssertionError(f"Oversized chunks in {path}: {oversized}")
        if any(
            chunk.body_token_count > settings.chunk_body_max_tokens
            or chunk.prefix_overlap_token_count
            > settings.chunk_overlap_max_tokens
            or chunk.suffix_overlap_token_count
            > settings.chunk_overlap_max_tokens
            or chunk.token_count > settings.chunk_total_max_tokens
            for chunk in chunks
        ):
            raise AssertionError(f"Token budget metadata exceeded for {path}")

        if any(re.match(r"^#{1,6}\s", chunk.text.lstrip()) for chunk in chunks):
            raise AssertionError(f"Heading duplicated in text payload for {path}")

        total_chunks += len(chunks)
        body_token_counts.extend(chunk.body_token_count for chunk in chunks)
        total_token_counts.extend(chunk.token_count for chunk in chunks)

    sorted_body = sorted(body_token_counts)
    p95_index = min(len(sorted_body) - 1, int(len(sorted_body) * 0.95))
    print(
        "Corpus Markdown chunking: "
        f"tokenizer={tokenizer_label} total={total_chunks} "
        f"body_median={median(sorted_body):.1f} "
        f"body_p95={sorted_body[p95_index]} body_max={max(sorted_body)} "
        f"assembled_max={max(total_token_counts)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-tokenizer",
        action="store_true",
        help="Use the configured Hugging Face Qwen3 tokenizer for corpus checks.",
    )
    args = parser.parse_args()
    corpus_tokenizer = CHAR_TOKENIZER
    tokenizer_label = "character-test-double"
    if args.real_tokenizer:
        corpus_tokenizer = VectorStore(settings).chunk_tokenizer
        tokenizer_label = settings.chunk_tokenizer_model

    assert_corpus(corpus_tokenizer, tokenizer_label=tokenizer_label)
    assert_synthetic_markdown()
    assert_overlap_budget()
    assert_heading_boundaries()
    assert_yaml_frontmatter()
    assert_indented_prose_is_text()
    assert_long_paragraph_uses_sentence_boundaries()
    assert_metadata_filter()
    assert_gfm_like_markdown()
    assert_token_aware_budget()


if __name__ == "__main__":
    main()
