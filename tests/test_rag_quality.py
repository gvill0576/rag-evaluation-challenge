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
        answer_lower = result["answer"].lower()
        out_of_scope_phrases = [
            "don't have",
            "unrelated",
            "not related",
            "cannot answer",
            "no information",
            "context provided",
            "real-time",
            "not available"
        ]
        assert any(phrase in answer_lower for phrase in out_of_scope_phrases), \
            "Expected model to indicate the question is out of scope"

    def test_suite_pass_rate(self, pipeline):
        results = run_test_suite(pipeline)
        assert results["pass_rate"] >= 0.75, \
            f"Pass rate {results['pass_rate']} below 0.75"
