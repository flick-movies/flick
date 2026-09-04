from src.hybrid.ml_reranker import (
    build_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Building ML reranker training dataset...")

    X, y, users_used = build_training_dataset(
        ratings=ratings,
        movies=movies,
    )

    print(f"Users used:       {users_used:,}")
    print(f"Training examples: {len(y):,}")
    print(f"Positive labels:   {(y == 1).sum():,}")
    print(f"Negative labels:   {(y == 0).sum():,}")

    print("\nTraining ranker...")

    ranker = train_ranker(X, y)

    save_ranker(ranker)

    print("\nSaved model to src/models/ml_reranker.joblib")


if __name__ == "__main__":
    main()