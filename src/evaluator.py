from src.metrics import measure_faithfulness, measure_relevance, measure_precision

TEST_CASES = [
    {"id": "TC001", "query": "What causes overfitting?", "category": "technical",
     "expected_keywords": ["memorize", "training", "data"],
     "min_scores": {"faithfulness": 7, "relevance": 7, "precision": 6}},
    {"id": "TC002", "query": "How do I prevent overfitting?", "category": "how-to",
     "expected_keywords": ["regularization", "dropout"],
     "min_scores": {"faithfulness": 7, "relevance": 7, "precision": 6}},
    {"id": "TC003", "query": "Explain backpropagation", "category": "conceptual",
     "expected_keywords": ["gradient", "weight"],
     "min_scores": {"faithfulness": 7, "relevance": 7, "precision": 6}},
    {"id": "TC004", "query": "What is a learning rate?", "category": "conceptual",
     "expected_keywords": ["step", "gradient"],
     "min_scores": {"faithfulness": 7, "relevance": 7, "precision": 6}},
    {"id": "TC005", "query": "How does cross-validation work?", "category": "technical",
     "expected_keywords": ["data", "model"],
     "min_scores": {"faithfulness": 7, "relevance": 7, "precision": 6}},
    {"id": "TC006", "query": "What is regularization?", "category": "how-to",
     "expected_keywords": ["overfitting", "model"],
     "min_scores": {"faithfulness": 7, "relevance": 6, "precision": 6}},
    {"id": "TC007", "query": "Explain gradient descent", "category": "conceptual",
     "expected_keywords": ["weight", "error"],
     "min_scores": {"faithfulness": 7, "relevance": 7, "precision": 6}},
    {"id": "TC008", "query": "What's the weather today?", "category": "out-of-scope",
     "max_scores": {"relevance": 4}},
]

def evaluate_response(query: str, context: str, answer: str, documents: list) -> dict:
    faith = measure_faithfulness(context, answer)
    rel = measure_relevance(query, answer)
    prec = measure_precision(query, documents)
    overall = faith["score"] * 0.4 + rel["score"] * 0.4 + prec["score"] * 0.2

    return {
        "metrics": {
            "faithfulness": faith["score"],
            "relevance": rel["score"],
            "precision": prec["score"],
            "overall": round(overall, 1)
        },
        "details": {"faithfulness": faith, "relevance": rel, "precision": prec}
    }

def run_test_suite(pipeline, test_cases: list = TEST_CASES) -> dict:
    results = []
    category_scores = {}

    for case in test_cases:
        response = pipeline.query(case["query"])
        evaluation = evaluate_response(
            case["query"], response["context"], response["answer"], response["documents"]
        )

        passed = True
        failures = []

        if "min_scores" in case:
            for metric, min_val in case["min_scores"].items():
                actual = evaluation["metrics"].get(metric, 0)
                if actual < min_val:
                    passed = False
                    failures.append(f"{metric}: {actual} < {min_val}")

        if "max_scores" in case:
            for metric, max_val in case["max_scores"].items():
                actual = evaluation["metrics"].get(metric, 10)
                if actual > max_val:
                    passed = False
                    failures.append(f"{metric}: {actual} > {max_val}")

        results.append({
            "id": case["id"],
            "query": case["query"],
            "category": case.get("category"),
            "passed": passed,
            "failures": failures,
            "metrics": evaluation["metrics"]
        })

        cat = case.get("category", "general")
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(evaluation["metrics"]["overall"])

    passed_count = sum(1 for r in results if r["passed"])

    return {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results),
        "category_averages": {
            cat: round(sum(s) / len(s), 1) for cat, s in category_scores.items()
        },
        "results": results
    }
