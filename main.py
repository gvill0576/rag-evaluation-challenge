import sys
from src.pipeline import RAGPipeline
from src.evaluator import evaluate_response, run_test_suite
from src.tracker import QualityTracker
from src.cache import EvaluationCache
from src.costs import CostTracker

def cmd_evaluate(query: str):
    print(f"\nEvaluating: {query}")
    pipeline = RAGPipeline()
    result = pipeline.query(query)
    evaluation = evaluate_response(query, result["context"], result["answer"], result["documents"])
    scores = evaluation["metrics"]
    print(f"Faithfulness: {scores['faithfulness']}/10")
    print(f"Relevance:    {scores['relevance']}/10")
    print(f"Precision:    {scores['precision']}/10")
    print(f"Overall:      {scores['overall']}/10")

def cmd_suite():
    print("\nRunning full test suite...")
    pipeline = RAGPipeline()
    results = run_test_suite(pipeline)
    for r in results["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']} - {r['query'][:40]} | Overall: {r['metrics']['overall']}")
        if r["failures"]:
            for f in r["failures"]:
                print(f"       Failed: {f}")
    print(f"\nTotal: {results['passed']}/{results['total']} passed ({results['pass_rate']*100:.0f}%)")
    print("\nCategory averages:")
    for cat, avg in results["category_averages"].items():
        print(f"  {cat}: {avg}/10")
    tracker = QualityTracker()
    tracker.log_evaluation(results)
    print("\nResults logged to data/quality_history.jsonl")

def cmd_trends():
    tracker = QualityTracker()
    history = tracker.get_history(days=30)
    print(f"\nHistory entries (last 30 days): {len(history)}")
    for metric in ["technical", "how-to", "conceptual", "out-of-scope"]:
        trend = tracker.get_trend(metric)
        print(f"  {metric}: {trend['trend']} (change: {trend.get('change', 'N/A')})")

def cmd_cache_stats():
    cache = EvaluationCache()
    stats = cache.stats()
    print(f"\nCache stats:")
    print(f"  Hits:      {stats['hits']}")
    print(f"  Misses:    {stats['misses']}")
    print(f"  Hit rate:  {stats['hit_rate']}")
    print(f"  Files:     {stats['cached_files']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [evaluate|suite|trends|cache]")
        print("  evaluate 'your question here'")
        print("  suite")
        print("  trends")
        print("  cache")
        sys.exit(1)

    command = sys.argv[1]
    if command == "evaluate" and len(sys.argv) > 2:
        cmd_evaluate(sys.argv[2])
    elif command == "suite":
        cmd_suite()
    elif command == "trends":
        cmd_trends()
    elif command == "cache":
        cmd_cache_stats()
    else:
        print(f"Unknown command: {command}")
