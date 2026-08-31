# src/test_evaluator.py

from pathlib import Path

import pandas as pd

from src.evaluation.ranking import evaluate_pairwise_accuracy


def main() -> None:

    project_root = Path(__file__).resolve().parent.parent.parent

    ratings_path = project_root / "data" / "ratings.csv"
    movies_path = project_root / "data" / "movies.csv"

    print("=" * 70)
    print("Testing ranking evaluator")
    print("=" * 70)

    print(f"Ratings: {ratings_path}")
    print(f"Movies:  {movies_path}")
    print()


    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {ratings_path}"
        )

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Movies file not found: {movies_path}"
        )

    print("Loading data...")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    print(f"Ratings loaded: {len(ratings):,}")
    print(f"Movies loaded:  {len(movies):,}")
    print()


    required_rating_columns = {
        "userId",
        "movieId",
        "rating",
    }

    missing_rating_columns = (
        required_rating_columns - set(ratings.columns)
    )

    if missing_rating_columns:
        raise ValueError(
            "Ratings is missing columns: "
            f"{sorted(missing_rating_columns)}"
        )

    required_movie_columns = {
        "movieId",
    }

    missing_movie_columns = (
        required_movie_columns - set(movies.columns)
    )

    if missing_movie_columns:
        raise ValueError(
            "Movies is missing columns: "
            f"{sorted(missing_movie_columns)}"
        )



    print("Running evaluator...")
    print("This may take a while because matrix factorization")
    print("is trained on the historical ratings.")
    print()

    try:
        results = evaluate_pairwise_accuracy(
            ratings=ratings,
            movies=movies,
        )
    except Exception:
        print()
        print("EVALUATOR FAILED")
        print("=" * 70)
        raise



    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Users evaluated:  {results.users_evaluated:,}"
    )

    print(
        f"Pairs evaluated:  {results.pairs_evaluated:,}"
    )

    print()
    print("Pairwise accuracy:")
    print(
        f"  Movie Average Baseline: "
        f"{results.baseline_accuracy:.4f} "
        f"({results.baseline_accuracy * 100:.2f}%)"
    )

    print(
        f"  Genre Heuristic:         "
        f"{results.heuristic_accuracy:.4f} "
        f"({results.heuristic_accuracy * 100:.2f}%)"
    )

    print(
        f"  ML Reranker:             "
        f"{results.ml_accuracy:.4f} "
        f"({results.ml_accuracy * 100:.2f}%)"
    )

    print(
        f"  Matrix Factorization:    "
        f"{results.matrix_factorization_accuracy:.4f} "
        f"({results.matrix_factorization_accuracy * 100:.2f}%)"
    )

    print()
    print("=" * 70)


    scores = {
        "Movie Average Baseline": results.baseline_accuracy,
        "Genre Heuristic": results.heuristic_accuracy,
        "ML Reranker": results.ml_accuracy,
        "Matrix Factorization": results.matrix_factorization_accuracy,
    }

    ranked_models = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print("Ranking:")
    print()

    for position, (name, score) in enumerate(
        ranked_models,
        start=1,
    ):
        print(
            f"  {position}. {name:<25} "
            f"{score * 100:.2f}%"
        )

    print()



    baseline = results.baseline_accuracy

    print("Improvement over movie-average baseline:")
    print()

    for name, score in ranked_models:
        if name == "Movie Average Baseline":
            continue

        improvement = score - baseline

        print(
            f"  {name:<25} "
            f"{improvement * 100:+.2f} percentage points"
        )

    print()
    print("Evaluator completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
