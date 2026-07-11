# RAG Evaluation Set — RAGBench / cuad

Long-document RAG benchmark. Unlike HotpotQA (single Wikipedia intro paragraphs),
these documents are multi-paragraph and long enough to actually exercise your
chunking strategy.

## Contents
- `corpus.jsonl`  — 102 documents. One JSON per line: `{id, title, text}`.
- `queries.jsonl` — 500 queries. One JSON per line:
  `{id, question, answer, gold_doc_ids, context_doc_ids, utilized_doc_ids,
     relevance_score, utilization_score, completeness_score, adherence_score}`.
- `../data/docs/RAGBench/` — the same 102 documents as individual `.md` files
  with frontmatter, used by the Compose ingest pipeline.

## Document length
median 3863 words/doc; corpus ranges 123–33948 words. Real multi-paragraph docs → chunking matters.

## Ground truth
- **Retrieval:** `gold_doc_ids` = documents containing a question-relevant sentence
  (derived from RAGBench `all_relevant_sentence_keys`). Compute recall@k / precision@k / MRR.
  412/500 queries have >=1 gold doc (some RAGBench items are deliberately
  low-relevance — useful for testing whether your system abstains).
- **Generation:** `answer` = reference response. Score faithfulness/correctness against it.
- **TRACe scores** (RAGBench's own labels) carried through as metadata for reference:
  `relevance_score`, `utilization_score`, `completeness_score`, `adherence_score`.
- `context_doc_ids` = the passages this question was originally given (avg 1.0/q);
  `utilized_doc_ids` = passages the reference answer actually used.

## Quick evaluation sketch
```python
import json
corpus  = [json.loads(l) for l in open("corpus.jsonl", encoding="utf-8")]
queries = [json.loads(l) for l in open("queries.jsonl", encoding="utf-8")]
# 1) chunk + index each corpus doc's text
# 2) retrieved = retrieve(q["question"], k=5) -> doc ids
#    recall = len(set(retrieved) & set(q["gold_doc_ids"])) / max(len(q["gold_doc_ids"]),1)
# 3) answer = your_rag(q["question"]); compare to q["answer"]
```

Source: RAGBench (Friel et al., 2024), config `cuad`, `test` split, first 500 items.
