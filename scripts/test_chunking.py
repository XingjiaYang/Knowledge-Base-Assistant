from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.vector_store import VectorStore, chunk_markdown


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
    chunks = chunk_markdown(sample, chunk_size=180, overlap=30)
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
    sample = "# Overlap\n\n" + " ".join(f"word{idx}" for idx in range(80))
    chunks = chunk_markdown(sample, chunk_size=120, overlap=30)
    text_chunks = [chunk for chunk in chunks if chunk.content_type == "text"]
    if len(text_chunks) < 2:
        raise AssertionError("Expected long text to split into multiple chunks.")

    oversized = [len(chunk.text) for chunk in text_chunks if len(chunk.text) > 120]
    if oversized:
        raise AssertionError(f"Text chunks exceeded chunk_size: {oversized}")

    first_tail = text_chunks[0].text.split()[-1]
    if first_tail not in text_chunks[1].text:
        raise AssertionError("Text overlap should carry boundary context forward.")

    print("Overlap budget -> ok")


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
    chunks = chunk_markdown(sample, chunk_size=240, overlap=20)
    if not any(chunk.content_type == "table" for chunk in chunks):
        raise AssertionError("GFM-style tables should be parsed as table blocks.")

    combined_text = "\n".join(chunk.text for chunk in chunks)
    if "~~legacy~~" not in combined_text:
        raise AssertionError("Strikethrough text should remain in chunk payload.")
    if "[x] Keep task list text" not in combined_text:
        raise AssertionError("Task-list text should remain retrievable.")

    print("GFM-like Markdown -> ok")


def main() -> None:
    total_chunks = 0

    for path in sorted(settings.docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise AssertionError(f"No chunks produced for {path}")

        if text.lstrip().startswith("#") and not any(chunk.h1 for chunk in chunks):
            raise AssertionError(f"Heading metadata missing in {path}")

        oversized = [
            len(chunk.text)
            for chunk in chunks
            if len(chunk.text) > settings.chunk_size
        ]
        if oversized:
            raise AssertionError(f"Oversized chunks in {path}: {oversized}")

        if any(chunk.text.lstrip().startswith("#") for chunk in chunks):
            raise AssertionError(f"Heading duplicated in text payload for {path}")

        total_chunks += len(chunks)
        print(f"{path.name}: {len(chunks)} chunks")

    print(f"Total chunks: {total_chunks}")
    assert_synthetic_markdown()
    assert_overlap_budget()
    assert_metadata_filter()
    assert_gfm_like_markdown()


if __name__ == "__main__":
    main()
