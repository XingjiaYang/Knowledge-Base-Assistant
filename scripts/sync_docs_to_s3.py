import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.s3_documents import S3DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local Markdown documents to the configured S3 bucket."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=settings.docs_dir,
        help="Local Markdown directory to sync. Defaults to DOCS_DIR.",
    )
    parser.add_argument(
        "--delete-removed",
        action="store_true",
        help="Delete S3 objects whose Markdown files no longer exist locally.",
    )
    args = parser.parse_args()

    result = S3DocumentStore(settings).sync_local_directory(
        args.docs_dir,
        delete_removed=args.delete_removed,
    )
    print(
        "S3 docs sync complete: "
        f"local_files={result['local_files']} remote_files={result['remote_files']} "
        f"uploaded={result['uploaded']} skipped={result['skipped']} "
        f"deleted={result['deleted']}"
    )


if __name__ == "__main__":
    main()
