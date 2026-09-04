from src.evaluation.ranking import (
    PairwiseEvaluation,
    UserPairwiseResult,
)


def test_user_pairwise_result_stores_metrics():
    result = UserPairwiseResult(
        user_id=23,
        rating_count=100,
        pairs_evaluated=50,
        baseline_accuracy=0.60,
        heuristic_accuracy=0.62,
        ml_accuracy=0.68,
        matrix_factorization_accuracy=0.66,
    )
    assert result.user_id == 23
    assert result.rating_count == 100
    assert result.pairs_evaluated == 50
    assert result.baseline_accuracy == 0.60
    assert result.heuristic_accuracy == 0.62
    assert result.ml_accuracy == 0.68
    assert result.matrix_factorization_accuracy == 0.66


def test_pairwise_evaluation_stores_user_results():
    user_results = [
        UserPairwiseResult(
            user_id=1,
            rating_count=50,
            pairs_evaluated=20,
            baseline_accuracy=0.60,
            heuristic_accuracy=0.55,
            ml_accuracy=0.70,
            matrix_factorization_accuracy=0.66,
        ),
        UserPairwiseResult(
            user_id=2,
            rating_count=80,
            pairs_evaluated=30,
            baseline_accuracy=0.65,
            heuristic_accuracy=0.60,
            ml_accuracy=0.75,
            matrix_factorization_accuracy=0.66,
        ),
    ]

    evaluation = PairwiseEvaluation(
        baseline_accuracy=0.625,
        heuristic_accuracy=0.575,
        ml_accuracy=0.725,
        matrix_factorization_accuracy=0.660,
        users_evaluated=2,
        pairs_evaluated=50,
        user_results=user_results,
    )

    assert evaluation.users_evaluated == 2
    assert evaluation.pairs_evaluated == 50
    assert len(evaluation.user_results) == 2

    assert evaluation.user_results[0].user_id == 1
    assert evaluation.user_results[1].user_id == 2


def test_pairwise_evaluation_user_count_matches_results():
    user_results = [
        UserPairwiseResult(
            user_id=1,
            rating_count=40,
            pairs_evaluated=10,
            baseline_accuracy=0.50,
            heuristic_accuracy=0.60,
            ml_accuracy=0.70,
            matrix_factorization_accuracy=0.66,
        ),
        UserPairwiseResult(
            user_id=2,
            rating_count=60,
            pairs_evaluated=15,
            baseline_accuracy=0.55,
            heuristic_accuracy=0.65,
            ml_accuracy=0.75,
            matrix_factorization_accuracy=0.66,
        ),
        UserPairwiseResult(
            user_id=3,
            rating_count=100,
            pairs_evaluated=25,
            baseline_accuracy=0.60,
            heuristic_accuracy=0.70,
            ml_accuracy=0.80,
            matrix_factorization_accuracy=0.66,
        ),
    ]

    evaluation = PairwiseEvaluation(
        baseline_accuracy=0.55,
        heuristic_accuracy=0.65,
        ml_accuracy=0.75,
        matrix_factorization_accuracy=0.660,
        users_evaluated=len(user_results),
        pairs_evaluated=50,
        user_results=user_results,
    )

    assert evaluation.users_evaluated == len(
        evaluation.user_results
    )


def test_pair_counts_match_user_results():
    user_results = [
        UserPairwiseResult(
            user_id=1,
            rating_count=40,
            pairs_evaluated=10,
            baseline_accuracy=0.50,
            heuristic_accuracy=0.60,
            ml_accuracy=0.70,
            matrix_factorization_accuracy=0.66
        ),
        UserPairwiseResult(
            user_id=2,
            rating_count=60,
            pairs_evaluated=15,
            baseline_accuracy=0.55,
            heuristic_accuracy=0.65,
            ml_accuracy=0.75,
            matrix_factorization_accuracy=0.66
        ),
    ]

    total_pairs = sum(
        result.pairs_evaluated
        for result in user_results
    )

    evaluation = PairwiseEvaluation(
        baseline_accuracy=0.525,
        heuristic_accuracy=0.625,
        ml_accuracy=0.725,
        matrix_factorization_accuracy=0.660,
        users_evaluated=2,
        pairs_evaluated=total_pairs,
        user_results=user_results,
    )

    assert evaluation.pairs_evaluated == sum(
        result.pairs_evaluated
        for result in evaluation.user_results
    )