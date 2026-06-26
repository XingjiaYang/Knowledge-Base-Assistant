import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.code_indexer import CodeIndexer, discover_code_repositories
from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index source code files into Qdrant and PostgreSQL.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Single source tree to index. Defaults to discovered code repositories.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Index all repositories discovered under CODE_ROOT_DIR.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Repository id or folder name to index. Can be passed more than once.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate code Qdrant collections and clear code PostgreSQL tables.",
    )
    args = parser.parse_args()

    indexer = CodeIndexer(settings)
    repositories = discover_code_repositories(settings)
    selected_repositories = repositories
    if args.repo:
        requested = {value.strip() for value in args.repo if value.strip()}
        selected_repositories = [
            repository
            for repository in repositories
            if repository.id in requested or repository.name in requested
        ]
        missing = requested - {
            value
            for repository in selected_repositories
            for value in (repository.id, repository.name)
        }
        if missing:
            raise SystemExit(f"Unknown code repository: {', '.join(sorted(missing))}")

    if args.source_dir is not None:
        stats = indexer.index_source_tree(args.source_dir, recreate=args.recreate)
        repo_count = 1 if stats.files else 0
    else:
        stats = indexer.index_repositories(
            selected_repositories,
            recreate=args.recreate,
        )
        repo_count = len(selected_repositories)
    print(
        "Indexed code: "
        f"repos={repo_count} files={stats.files} functions={stats.functions} "
        f"call_edges={stats.call_edges}"
    )


if __name__ == "__main__":
    main()
