from pathlib import Path

from src.hybrid.ml_reranker import (
    build_ablation_training_dataset,
    save_ranker,
    train_ranker,
)
from src.load_data import load_movielens


EXPERIMENTS = {
    "mf_oldpersonal_quality_popularity": [
        "collaborative",
        "personal",
        "quality",
        "popularity",
    ],
    "mf_quality_popularity": [
        "collaborative",
        "quality",
        "popularity",
    ],
    "mf_content": [
        "collaborative",
        "content",
    ],
    "mf_oldpersonal": [
        "collaborative",
        "personal",
    ],
    "mf_oldpersonal_content": [
        "collaborative",
        "personal",
        "content",
    ],
    "all_5": [
        "collaborative",
        "personal",
        "content",
        "quality",
        "popularity",
    ],
}


def main() -> None:
    ratings, movies = load_movielens("data")

    output_dir = Path("src/models/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Week 3 Ablation Training")
    print("=" * 72)

    print(f"Ratings: {len(ratings):,}")
    print(f"Users:   {ratings['userId'].nunique():,}")
    print()

    for experiment_name, features in EXPERIMENTS.items():
        print("=" * 72)
        print(f"Experiment: {experiment_name}")
        print(f"Features:   {', '.join(features)}")
        print("=" * 72)

        X, y, users_used = build_ablation_training_dataset(
            ratings=ratings,
            movies=movies,
            features=features,
        )

        print(f"Users used:        {users_used:,}")
        print(f"Training examples: {len(y):,}")
        print(f"Feature count:     {X.shape[1]}")
        print(f"Positive labels:   {(y == 1).sum():,}")
        print(f"Negative labels:   {(y == 0).sum():,}")

        print("\nTraining reranker...")

        ranker = train_ranker(X, y)

        print("Coefficients:")

        for feature, coefficient in zip(
            features,
            ranker.model.coef_[0],
        ):
            print(
                f"  {feature:<16} "
                f"{coefficient:+.6f}"
            )

        model_path = (
            output_dir /
            f"{experiment_name}.joblib"
        )

        save_ranker(
            ranker,
            path=str(model_path),
        )

        print(f"\nSaved: {model_path}")
        print()

    print("=" * 72)
    print("All six ablation models trained.")
    print("=" * 72)


if __name__ == "__main__":
    main()