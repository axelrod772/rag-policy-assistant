"""
RAGAS evaluation: Faithfulness, Answer Relevancy, Context Precision.
Run from project root: python -m scripts.eval_ragas --questions_file eval/sample_questions.json --output eval/ragas_results.json
"""
import argparse
import json
from pathlib import Path

from src.retrieval import HybridRetriever
from src.generate import get_generator, build_prompt, generate_answer

ROOT = Path(__file__).resolve().parent.parent
IDX_DIR = ROOT / "artifacts" / "index"


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "questions" in data:
        return data["questions"]
    return [data]


def run_rag(questions: list[dict], retr: HybridRetriever, gen) -> list[dict]:
    """Run RAG for each question; return list with question, answer, contexts, ground_truth if present."""
    results = []
    for item in questions:
        q = item.get("question") or item.get("query") or ""
        if not q.strip():
            continue
        hits = retr.search(q, k=5, alpha=0.5)
        prompt = build_prompt(q, hits)
        answer = generate_answer(gen, prompt, max_new_tokens=160, temperature=0.0)
        contexts = [c[1] for c in hits]
        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth") or item.get("answer"),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", type=str, default="eval/sample_questions.json")
    parser.add_argument("--output", type=str, default="eval/ragas_results.json")
    parser.add_argument("--index_dir", type=str, default=None)
    args = parser.parse_args()

    index_dir = args.index_dir or str(IDX_DIR)
    if not (Path(index_dir) / "faiss.index").exists():
        print("Index not found. Run: python -m scripts.build_index")
        return

    questions_path = Path(args.questions_file)
    if not questions_path.is_absolute():
        questions_path = ROOT / questions_path
    if not questions_path.exists():
        # Create sample file
        questions_path.parent.mkdir(parents=True, exist_ok=True)
        sample = [
            {"question": "What is the waiting period for pre-existing conditions?", "ground_truth": "12 months."},
            {"question": "Who is eligible for coverage?", "ground_truth": "Full-time employees and dependents after 90 days."},
        ]
        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)
        print(f"Created sample questions at {questions_path}")

    questions = load_questions(str(questions_path))
    retr = HybridRetriever()
    retr.load(index_dir)
    gen = get_generator()
    rag_results = run_rag(questions, retr, gen)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        # RAGAS expects columns: question, answer, contexts, ground_truth
        ds = Dataset.from_list([
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"] or "",
            }
            for r in rag_results
        ])
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        # Handle both dict and EvaluationResult-style result
        if hasattr(result, "__dict__"):
            res_dict = getattr(result, "scores", result.__dict__)
        else:
            res_dict = dict(result) if hasattr(result, "items") else {}
        scores = {
            "faithfulness": res_dict.get("faithfulness", 0.0),
            "answer_relevancy": res_dict.get("answer_relevancy", 0.0),
            "context_precision": res_dict.get("context_precision", 0.0),
        }
    except Exception as e:
        scores = {"error": str(e), "faithfulness": None, "answer_relevancy": None, "context_precision": None}

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "rag_results": rag_results}, f, indent=2)
    print(f"Results written to {out_path}")
    print("Scores:", scores)


if __name__ == "__main__":
    main()
