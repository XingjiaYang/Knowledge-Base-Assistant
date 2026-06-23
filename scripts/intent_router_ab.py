from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.intent_router import IntentDecision, IntentRouter
from app.vector_store import VectorStore


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "intent_router_cases.jsonl"


@dataclass(frozen=True)
class CaseMessage:
    role: str
    content: str


@dataclass(frozen=True)
class IntentCase:
    id: str
    category: str
    question: str
    expected_use_rag: bool
    history: tuple[CaseMessage, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class Variant:
    name: str
    config: Settings
    use_embedding: bool


@dataclass(frozen=True)
class CaseResult:
    case: IntentCase
    decision: IntentDecision

    @property
    def correct(self) -> bool:
        return self.decision.use_rag == self.case.expected_use_rag


class FakeIntentEmbedder:
    def encode(
        self,
        texts: str | Sequence[str],
        normalize_embeddings: bool = True,
    ) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        lowered = text.lower()
        rag_markers = (
            "local knowledge base",
            "indexed documents",
            "document evidence",
            "local corpus",
            "local docs",
            "知识库",
            "本地文档",
            "引用资料",
            "文档",
            "家是本",
            "jiashiben",
            "jia shi ben",
            "朱剑秋",
            "zhu jianqiu",
            "勇哥",
            "yongge",
            "巨大历史机遇",
            "巨大历史鲫鱼",
            "giant historical opportunity",
            "giant historical crucian carp",
            "b站",
            "bilibili",
            "菜单",
            "pricing controversies",
            "财务",
            "评论",
            "reviews",
            "doubao",
            "slogan",
        )
        direct_markers = (
            "general chat",
            "write or rewrite",
            "without documents",
            "translate this text",
            "programming question",
            "greet the user",
            "闲聊",
            "直接聊天",
            "问候",
            "感谢",
            "翻译",
            "写作",
            "简历",
            "不用知识库",
            "不要检索",
            "postgresql",
            "python",
            "fastapi",
            "javascript",
            "transformer",
            "normal restaurant",
        )
        rag = any(marker in lowered for marker in rag_markers)
        direct = any(marker in lowered for marker in direct_markers)
        if rag and not direct:
            return [1.0, 0.0]
        if direct and not rag:
            return [0.0, 1.0]
        if rag and direct:
            return [0.55, 0.45]
        return [0.5, 0.5]


def load_cases(path: Path) -> list[IntentCase]:
    cases: list[IntentCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            record = json.loads(stripped)
            history = tuple(
                CaseMessage(
                    role=str(message["role"]),
                    content=str(message["content"]),
                )
                for message in record.get("history", [])
            )
            cases.append(
                IntentCase(
                    id=str(record["id"]),
                    category=str(record.get("category", "uncategorized")),
                    question=str(record["question"]),
                    expected_use_rag=bool(record["expected_use_rag"]),
                    history=history,
                    summary=str(record.get("summary", "")),
                )
            )
            if not cases[-1].question.strip():
                raise ValueError(f"{path}:{line_number} has an empty question.")
    if not cases:
        raise ValueError(f"No cases loaded from {path}.")
    return cases


def parse_threshold_variant(spec: str, base: Settings) -> Variant:
    try:
        name, values = spec.split("=", maxsplit=1)
        rag_threshold, direct_threshold, margin = (
            float(value.strip()) for value in values.split(",", maxsplit=2)
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Variant must be NAME=RAG_THRESHOLD,DIRECT_THRESHOLD,MARGIN."
        ) from exc

    if not name.strip():
        raise argparse.ArgumentTypeError("Variant name must not be empty.")
    return Variant(
        name=name.strip(),
        config=replace(
            base,
            intent_embedding_rag_threshold=rag_threshold,
            intent_embedding_direct_threshold=direct_threshold,
            intent_embedding_margin=margin,
        ),
        use_embedding=True,
    )


def parse_model_variant(spec: str, base: Settings) -> Variant:
    try:
        name, values = spec.split("=", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Model variant must be NAME=MODEL[,CLASSIFICATION_TASK[,TRUST_REMOTE_CODE]]."
        ) from exc

    parts = [part.strip() for part in values.split(",")]
    model = parts[0] if parts else ""
    if not name.strip() or not model:
        raise argparse.ArgumentTypeError("Model variant name and model must be set.")

    is_jina = "jina" in model.lower()
    task = (
        parts[1]
        if len(parts) >= 2
        else base.embedding_classification_task if is_jina else ""
    )
    trust_remote_code = (
        _parse_bool(parts[2])
        if len(parts) >= 3 and parts[2]
        else base.embedding_trust_remote_code if is_jina else False
    )
    return Variant(
        name=name.strip(),
        config=replace(
            base,
            embedding_model=model,
            embedding_classification_task=task,
            embedding_trust_remote_code=trust_remote_code,
        ),
        use_embedding=True,
    )


def default_variants(config: Settings) -> list[Variant]:
    return [
        Variant("A_keyword_only", config, use_embedding=False),
        Variant("B_embedding_current", config, use_embedding=True),
    ]


def evaluate_variant(
    variant: Variant,
    cases: Sequence[IntentCase],
    embedder: object | None,
) -> list[CaseResult]:
    router = IntentRouter(
        variant.config,
        embedder=embedder if variant.use_embedding else None,
        llm_client=None,
    )
    return [
        CaseResult(
            case=case,
            decision=router.route(case.question, case.history, case.summary),
        )
        for case in cases
    ]


def embedder_cache_key(config: Settings) -> tuple[str, str, bool]:
    return (
        config.embedding_model,
        config.embedding_classification_task,
        config.embedding_trust_remote_code,
    )


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def summarize(results: Sequence[CaseResult]) -> dict[str, float | int]:
    total = len(results)
    correct = sum(1 for result in results if result.correct)
    predicted_rag = sum(1 for result in results if result.decision.use_rag)
    expected_rag = sum(1 for result in results if result.case.expected_use_rag)
    true_rag = sum(
        1
        for result in results
        if result.decision.use_rag and result.case.expected_use_rag
    )
    true_direct = sum(
        1
        for result in results
        if not result.decision.use_rag and not result.case.expected_use_rag
    )
    false_rag = sum(
        1
        for result in results
        if result.decision.use_rag and not result.case.expected_use_rag
    )
    false_direct = sum(
        1
        for result in results
        if not result.decision.use_rag and result.case.expected_use_rag
    )
    embedding_decisions = sum(
        1 for result in results if result.decision.route.startswith("embedding_")
    )
    keyword_decisions = sum(
        1 for result in results if result.decision.route.startswith("keyword_")
    )
    fallback_decisions = sum(
        1 for result in results if result.decision.route.startswith("fallback_")
    )
    rag_precision = true_rag / predicted_rag if predicted_rag else 0.0
    rag_recall = true_rag / expected_rag if expected_rag else 0.0
    direct_total = total - expected_rag
    direct_recall = true_direct / direct_total if direct_total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "rag_precision": rag_precision,
        "rag_recall": rag_recall,
        "direct_recall": direct_recall,
        "false_rag": false_rag,
        "false_direct": false_direct,
        "keyword_decisions": keyword_decisions,
        "embedding_decisions": embedding_decisions,
        "fallback_decisions": fallback_decisions,
    }


def summarize_by_category(
    results: Sequence[CaseResult],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case.category, []).append(result)
    return {
        category: summarize(grouped[category])
        for category in sorted(grouped)
    }


def compare(
    control: Sequence[CaseResult],
    treatment: Sequence[CaseResult],
) -> dict[str, int]:
    control_by_id = {result.case.id: result for result in control}
    treatment_by_id = {result.case.id: result for result in treatment}
    shared_ids = sorted(set(control_by_id) & set(treatment_by_id))
    fixed = 0
    regressions = 0
    changed = 0
    direct_savings = 0
    rag_misses = 0
    for case_id in shared_ids:
        a = control_by_id[case_id]
        b = treatment_by_id[case_id]
        if a.decision.use_rag != b.decision.use_rag:
            changed += 1
        if not a.correct and b.correct:
            fixed += 1
        if a.correct and not b.correct:
            regressions += 1
        if (
            a.decision.use_rag
            and not b.decision.use_rag
            and not b.case.expected_use_rag
        ):
            direct_savings += 1
        if not b.decision.use_rag and b.case.expected_use_rag:
            rag_misses += 1
    return {
        "changed": changed,
        "fixed": fixed,
        "regressions": regressions,
        "direct_savings": direct_savings,
        "rag_misses": rag_misses,
    }


def print_summary_table(results_by_variant: dict[str, list[CaseResult]]) -> None:
    headers = (
        "variant",
        "acc",
        "rag_p",
        "rag_r",
        "direct_r",
        "false_rag",
        "false_direct",
        "keyword",
        "embedding",
        "fallback",
    )
    print(" ".join(header.rjust(12) for header in headers))
    for name, results in results_by_variant.items():
        stats = summarize(results)
        row = (
            name,
            f"{stats['accuracy']:.3f}",
            f"{stats['rag_precision']:.3f}",
            f"{stats['rag_recall']:.3f}",
            f"{stats['direct_recall']:.3f}",
            str(stats["false_rag"]),
            str(stats["false_direct"]),
            str(stats["keyword_decisions"]),
            str(stats["embedding_decisions"]),
            str(stats["fallback_decisions"]),
        )
        print(" ".join(value.rjust(12) for value in row))


def print_category_table(results_by_variant: dict[str, list[CaseResult]]) -> None:
    print("\nBy category")
    headers = (
        "variant",
        "category",
        "n",
        "acc",
        "rag_r",
        "direct_r",
        "false_rag",
        "false_direct",
    )
    print(" ".join(header.rjust(16) for header in headers))
    for name, results in results_by_variant.items():
        for category, stats in summarize_by_category(results).items():
            row = (
                name,
                category,
                str(stats["total"]),
                f"{stats['accuracy']:.3f}",
                f"{stats['rag_recall']:.3f}",
                f"{stats['direct_recall']:.3f}",
                str(stats["false_rag"]),
                str(stats["false_direct"]),
            )
            print(" ".join(value.rjust(16) for value in row))


def print_disagreements(
    results_by_variant: dict[str, list[CaseResult]],
    baseline_name: str,
    max_cases: int,
) -> None:
    names = list(results_by_variant)
    baseline = {result.case.id: result for result in results_by_variant[baseline_name]}
    printed = 0
    for name in names:
        if name == baseline_name:
            continue
        print(f"\nDisagreements: {baseline_name} vs {name}")
        candidate = {result.case.id: result for result in results_by_variant[name]}
        for case_id, base_result in baseline.items():
            other = candidate[case_id]
            if base_result.decision.use_rag == other.decision.use_rag:
                continue
            expected = "rag" if base_result.case.expected_use_rag else "direct"
            base_pred = "rag" if base_result.decision.use_rag else "direct"
            other_pred = "rag" if other.decision.use_rag else "direct"
            print(
                f"- {case_id} expected={expected} "
                f"{baseline_name}={base_pred}/{base_result.decision.route} "
                f"{name}={other_pred}/{other.decision.route} "
                f"question={base_result.case.question}"
            )
            printed += 1
            if printed >= max_cases:
                return


def write_json_report(
    path: Path,
    results_by_variant: dict[str, list[CaseResult]],
) -> None:
    report = {
        "summary": {
            name: summarize(results) for name, results in results_by_variant.items()
        },
        "by_category": {
            name: summarize_by_category(results)
            for name, results in results_by_variant.items()
        },
        "comparisons": {},
        "cases": {},
    }
    names = list(results_by_variant)
    baseline_name = names[0]
    for name in names[1:]:
        report["comparisons"][f"{baseline_name}_vs_{name}"] = compare(
            results_by_variant[baseline_name],
            results_by_variant[name],
        )
    for name, results in results_by_variant.items():
        report["cases"][name] = [
            {
                "id": result.case.id,
                "category": result.case.category,
                "question": result.case.question,
                "expected_use_rag": result.case.expected_use_rag,
                "predicted_use_rag": result.decision.use_rag,
                "route": result.decision.route,
                "reason": result.decision.reason,
                "correct": result.correct,
            }
            for result in results
        ]
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an offline A/B evaluation for the intent router.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="JSONL file with labeled intent cases.",
    )
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use a deterministic fake embedder for fast harness checks.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=RAG,DIRECT,MARGIN",
        help=(
            "Add an embedding threshold variant, for example "
            "strict=0.46,0.48,0.10."
        ),
    )
    parser.add_argument(
        "--model-variant",
        action="append",
        default=[],
        metavar="NAME=MODEL[,TASK[,TRUST]]",
        help=(
            "Add an encoder variant. For old BGE without prompt/task kwargs, use "
            "old_bge=BAAI/bge-small-en-v1.5,,0."
        ),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--max-disagreements",
        type=int,
        default=25,
        help="Maximum disagreement rows to print.",
    )
    args = parser.parse_args()

    config = Settings()
    cases = load_cases(args.cases)
    variants = default_variants(config)
    variants.extend(parse_threshold_variant(spec, config) for spec in args.variant)
    variants.extend(parse_model_variant(spec, config) for spec in args.model_variant)

    fake_embedder = FakeIntentEmbedder() if args.fake_embedder else None
    embedder_cache: dict[tuple[str, str, bool], object] = {}

    def embedder_for(variant: Variant) -> object | None:
        if not variant.use_embedding:
            return None
        if fake_embedder is not None:
            return fake_embedder
        key = embedder_cache_key(variant.config)
        if key not in embedder_cache:
            embedder_cache[key] = VectorStore(variant.config)
        return embedder_cache[key]

    results_by_variant = {
        variant.name: evaluate_variant(variant, cases, embedder_for(variant))
        for variant in variants
    }

    print(f"Cases: {len(cases)}")
    print_summary_table(results_by_variant)
    print_category_table(results_by_variant)

    names = list(results_by_variant)
    baseline_name = names[0]
    for name in names[1:]:
        comparison = compare(results_by_variant[baseline_name], results_by_variant[name])
        print(
            f"\n{baseline_name} vs {name}: "
            f"changed={comparison['changed']} "
            f"fixed={comparison['fixed']} "
            f"regressions={comparison['regressions']} "
            f"direct_savings={comparison['direct_savings']} "
            f"rag_misses={comparison['rag_misses']}"
        )

    print_disagreements(results_by_variant, baseline_name, args.max_disagreements)

    if args.json_report:
        write_json_report(args.json_report, results_by_variant)
        print(f"\nWrote JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
