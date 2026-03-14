import pytest
from src.pipeline import RAGPipeline
from src.evaluator import evaluate_response, run_test_suite

MIN_FAITHFULNESS = 7
MIN_RELEVANCE = 7
MIN_PRECISION = 6

@pytest.fixture
def pipeline():
    return RAGPipeline()

class TestRAGQuality:
    def test_technical_query(self, pipeline):
        result = pipeline.query("What causes overfitting?")
        evaluation = evaluate_response(
            "What causes overfitting?",
            result["context"], result["answer"], result["documents"]
        )
        assert evaluation["metrics"]["faithfulness"] >= MIN_FAITHFULNESS
        assert evaluation["metrics"]["relevance"] >= MIN_RELEVANCE
        assert evaluation["metrics"]["precision"] >= MIN_PRECISION

    def test_out_of_scope(self, pipeline):
        result = pipeline.query("What's the weather?")
        evaluation = evaluate_response(
            "What's the weather?",
            result["context"], result["answer"], result["documents"]
        )
        assert evaluation["metrics"]["relevance"] <= 5 or \
               "don't have" in result["answer"].lower()

    def test_suite_pass_rate(self, pipeline):
        results = run_test_suite(pipeline)
        assert results["pass_rate"] >= 0.75, \
            f"Pass rate {results['pass_rate']} below 0.75"
