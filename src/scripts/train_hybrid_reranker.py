import numpy as np
from src.hybrid.content_adapter import build_content_model_from_frames

from src.hybrid.ml_reranker import (
    build_hybrid_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Building hybrid reranker training dataset...")

    X, y, users_used = build_hybrid_training_dataset(
        ratings=ratings,
        movies=movies,
    )

    print(f"Users used:        {users_used:,}")
    print(f"Training examples: {len(y):,}")
    print(f"Feature count:     {X.shape[1]}")
    print(f"Positive labels:   {(y == 1).sum():,}")
    print(f"Negative labels:   {(y == 0).sum():,}")

    print("\nTraining hybrid ranker...")

    ranker = train_ranker(X, y)

    print(
        "Coefficients:",
        ranker.model.coef_[0],
    )

    save_ranker(
        ranker,
        path="src/models/hybrid_v1_reranker.joblib",
    )

    print(
        "\nSaved model to "
        "src/models/hybrid_v1_reranker.joblib"
    )


if __name__ == "__main__":
    main()