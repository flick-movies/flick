from src.load_data import load_movielens
from src.evaluation.ranking import evaluate_pairwise_accuracy


def main() -> None:
    ratings, movies = load_movielens("data")

    result = evaluate_pairwise_accuracy(
        ratings=ratings,
        movies=movies,
    )

    print("Pairwise Ranking Evaluation")
    print("---------------------------")
    print(f"Users evaluated: {result.users_evaluated}")
    print(f"Test pairs: {result.pairs_evaluated}")
    print()
    print(f"Baseline accuracy:  {result.baseline_accuracy:.3%}")
    print(f"Heuristic accuracy: {result.heuristic_accuracy:.3%}")
    print(f"ML accuracy:        {result.ml_accuracy:.3%}")
    print(
        f"Difference:         "
        f"{result.ml_accuracy - result.heuristic_accuracy:+.3%}"
    )


if __name__ == "__main__":
    main()