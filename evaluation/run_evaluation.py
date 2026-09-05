"""Run automated RAG evaluation against the live app.

Usage:
    python evaluation/run_evaluation.py

The script imports the application's RAG chain, executes the golden questions,
and writes a JSON report. It does not expose API keys or write secrets.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_upgrades.evaluation import RAGEvaluator


def _run_sync_pipeline(question: str):
    from app_telegram_hybrid_bm25 import rag_chain
    if rag_chain is None:
        raise RuntimeError("RAG chain is not initialized. Start the application first or provide a custom pipeline.")
    result = asyncio.run(rag_chain.ainvoke({"input": question, "chat_history": []}))
    return {"answer": result.get("answer", ""), "documents": result.get("context", [])}


def main() -> int:
    dataset = os.getenv("RAG_EVAL_DATASET", "evaluation/golden_dataset.jsonl")
    output = os.getenv("RAG_EVAL_OUTPUT", "evaluation/results/latest.json")
    evaluator = RAGEvaluator(_run_sync_pipeline)
    records = evaluator.load_jsonl(dataset)
    report = evaluator.evaluate(records)
    evaluator.save_report(report, output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved evaluation report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
