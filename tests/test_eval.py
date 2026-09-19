"""Tests for retrieval evaluation utilities: metrics, RRF fusion, BM25."""

from src.eval.eval_harness import evaluate_retriever, recall_at_k
from src.eval.fusion import reciprocal_rank_fusion
from src.inference_pipeline.backends.bm25 import BM25Retriever, tokenize


class TestRecallAtK:
    """Test recall@k metric."""

    def test_all_relevant_found(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b"], ["a", "c", "d"], k=3) == 0.5

    def test_cutoff_excludes_later_hits(self):
        # "b" is relevant but only appears past k=1
        assert recall_at_k(["a", "b"], ["c", "b"], k=1) == 0.0

    def test_no_relevant_items(self):
        assert recall_at_k([], ["a", "b"], k=2) == 0.0


class TestReciprocalRankFusion:
    """Test RRF fusion of ranked lists."""

    def test_agreement_boosts_rank(self):
        # "x" is #1 in both lists, should come out on top
        ranking_a = ["x", "y", "z"]
        ranking_b = ["x", "z", "y"]
        fused = reciprocal_rank_fusion([ranking_a, ranking_b])
        assert fused[0] == "x"

    def test_union_of_ids_present(self):
        ranking_a = ["a", "b"]
        ranking_b = ["c", "d"]
        fused = reciprocal_rank_fusion([ranking_a, ranking_b])
        assert set(fused) == {"a", "b", "c", "d"}

    def test_single_ranking_preserves_order(self):
        ranking = ["a", "b", "c"]
        assert reciprocal_rank_fusion([ranking]) == ranking


class TestBM25Retriever:
    """Test the BM25 lexical retrieval baseline."""

    def _corpus(self):
        return [
            {"id": "1", "combined_text": "barbell bench press for chest strength"},
            {"id": "2", "combined_text": "bodyweight stretching routine for hamstrings"},
            {"id": "3", "combined_text": "dumbbell curl for biceps"},
        ]

    def test_retrieve_returns_ids(self):
        retriever = BM25Retriever(self._corpus())
        results = retriever.retrieve("barbell chest press", top_k=2)
        assert results[0] == "1"
        assert len(results) <= 2

    def test_retrieve_ranks_lexical_match_first(self):
        retriever = BM25Retriever(self._corpus())
        assert retriever.retrieve("dumbbell biceps curl", top_k=1) == ["3"]

    def test_tokenize_lowercases_and_strips_punctuation(self):
        assert tokenize("Barbell Bench-Press!") == ["barbell", "bench", "press"]


class TestEvaluateRetriever:
    """Test the generic retrieve_fn evaluation harness."""

    def test_perfect_retriever_scores_one(self):
        testset = [{"query": "q1", "relevant": ["a"]}, {"query": "q2", "relevant": ["b"]}]

        def retrieve_fn(query: str, top_k: int) -> list[str]:
            return {"q1": ["a"], "q2": ["b"]}[query][:top_k]

        metrics = evaluate_retriever(testset, retrieve_fn, top_k=1)
        assert metrics["recall@k"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["num_queries"] == 2

    def test_empty_retriever_scores_zero(self):
        testset = [{"query": "q1", "relevant": ["a"]}]

        def retrieve_fn(query: str, top_k: int) -> list[str]:
            return []

        metrics = evaluate_retriever(testset, retrieve_fn, top_k=5)
        assert metrics["recall@k"] == 0.0
        assert metrics["mrr"] == 0.0
