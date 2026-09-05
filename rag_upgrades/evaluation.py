"""Deterministic automated evaluation for retrieval and answer regressions.

The evaluator is framework-agnostic: inject a pipeline callable returning a dict
with `answer` and `documents`. Golden records may specify relevant source URLs,
course codes, required answer terms, and required citation sources.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("source", ""))


def _course_codes(doc: Any) -> set[str]:
    metadata = getattr(doc, "metadata", {}) or {}
    codes = metadata.get("course_codes", [])
    if isinstance(codes, str):
        return {codes.upper()}
    return {str(c).upper() for c in codes}


def _text(doc: Any) -> str:
    return str(getattr(doc, "page_content", doc))


def hit_rate(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = set(retrieved[:k])
    return 1.0 if top & relevant else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for item in retrieved[:k] if item in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def answer_keyword_coverage(answer: str, expected_terms: Sequence[str]) -> float:
    if not expected_terms:
        return 1.0
    answer_n = _normalise(answer)
    matched = sum(1 for term in expected_terms if _normalise(str(term)) in answer_n)
    return matched / len(expected_terms)


def citation_completeness(answer: str, required_sources: Sequence[str]) -> float:
    if not required_sources:
        return 1.0
    answer_n = _normalise(answer)
    matched = sum(1 for source in required_sources if _normalise(str(source)) in answer_n)
    return matched / len(required_sources)


def evaluate_retrieval(documents: Sequence[Any], relevant_sources: Sequence[str] = (), relevant_course_codes: Sequence[str] = (), ks: Sequence[int] = (3, 5, 10)) -> Dict[str, float]:
    docs = list(documents)
    source_keys = [_source(d) for d in docs]
    source_rel = {str(s) for s in relevant_sources if s}
    code_rel = {str(c).upper() for c in relevant_course_codes if c}

    if code_rel:
        ranked_relevant = [str(i) for i, d in enumerate(docs) if _course_codes(d) & code_rel]
        ranked_all = [str(i) for i in range(len(docs))]
        relevant = set(ranked_relevant)
        retrieved = ranked_all
        key_prefix = "course"
    else:
        relevant = source_rel
        retrieved = source_keys
        key_prefix = "source"

    result: Dict[str, float] = {}
    for k in ks:
        result[f"{key_prefix}_hit_rate@{k}"] = hit_rate(retrieved, relevant, k)
        result[f"{key_prefix}_precision@{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"{key_prefix}_recall@{k}"] = recall_at_k(retrieved, relevant, k)
        result[f"{key_prefix}_ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    result[f"{key_prefix}_mrr"] = reciprocal_rank(retrieved, relevant)
    return result


@dataclass
class EvalCaseResult:
    case_id: str
    question: str
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    answer_preview: str = ""
    retrieved_count: int = 0
    error: Optional[str] = None


class RAGEvaluator:
    """Run a golden set against an injected live or offline RAG pipeline."""

    def __init__(self, pipeline: Callable[[str], Dict[str, Any]], ks: Sequence[int] = (3, 5, 10)):
        self.pipeline = pipeline
        self.ks = tuple(ks)

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def evaluate(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        case_results: List[EvalCaseResult] = []
        for record in records:
            started = time.perf_counter()
            question = str(record.get("question", ""))
            case_id = str(record.get("id", question[:40]))
            try:
                result = self.pipeline(question) or {}
                if hasattr(result, "__await__"):
                    raise TypeError("RAGEvaluator requires a synchronous pipeline; wrap async pipelines before use")
                docs = list(result.get("documents", result.get("context", [])) or [])
                answer = str(result.get("answer", ""))
                metrics = evaluate_retrieval(
                    docs,
                    relevant_sources=record.get("relevant_sources", []),
                    relevant_course_codes=record.get("relevant_course_codes", []),
                    ks=self.ks,
                )
                metrics["answer_keyword_coverage"] = answer_keyword_coverage(
                    answer, record.get("expected_answer_terms", [])
                )
                metrics["citation_completeness"] = citation_completeness(
                    answer, record.get("required_citations", [])
                )
                case_results.append(EvalCaseResult(
                    case_id=case_id,
                    question=question,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    metrics=metrics,
                    answer_preview=answer[:500],
                    retrieved_count=len(docs),
                ))
            except Exception as exc:
                case_results.append(EvalCaseResult(
                    case_id=case_id,
                    question=question,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    error=f"{type(exc).__name__}: {exc}",
                ))

        metric_names = sorted({key for case in case_results for key in case.metrics})
        summary = {
            metric: round(sum(case.metrics.get(metric, 0.0) for case in case_results) / len(case_results), 4)
            if case_results else 0.0
            for metric in metric_names
        }
        summary["cases"] = len(case_results)
        summary["errors"] = sum(1 for case in case_results if case.error)
        summary["mean_latency_ms"] = round(
            sum(case.latency_ms for case in case_results) / len(case_results), 3
        ) if case_results else 0.0
        return {
            "summary": summary,
            "cases": [asdict(case) for case in case_results],
        }

    @staticmethod
    def save_report(report: Dict[str, Any], path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


def compare_reports(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    base = baseline.get("summary", {})
    new = candidate.get("summary", {})
    keys = sorted(set(base) & set(new) - {"cases", "errors"})
    return {
        key: {
            "baseline": base[key],
            "candidate": new[key],
            "delta": round(new[key] - base[key], 4),
        }
        for key in keys
        if isinstance(base[key], (int, float)) and isinstance(new[key], (int, float))
    }
