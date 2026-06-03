import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown documents into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before ingesting documents.",
    )
    args = parser.parse_args()

    store = VectorStore(settings)
    inserted = store.ingest_markdown_dir(recreate=args.recreate)
    if inserted == 0:
        print(f"No chunks generated from Markdown documents in {settings.docs_dir}")
        return
    print(f"Ingested {inserted} chunks into collection: {settings.collection_name}")


if __name__ == "__main__":
    main()
