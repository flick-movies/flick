from src.load_data import load_movielens
from src.ml_reranker import (
    build_training_dataset,
    train_ranker,
)


def main() -> None:
    ratings, movies = load_movielens("data")

    X, y, users_used = build_training_dataset(
        ratings=ratings,
        movies=movies,
    )

    print("Users used:", users_used)
    print("Training examples:", len(X))
    print("Feature matrix shape:", X.shape)

    print("Positive labels:", int(y.sum()))
    print("Negative labels:", int(len(y) - y.sum()))

    ranker = train_ranker(X, y)

    print()
    print("Learned coefficients:")
    print("Personal score:", ranker.model.coef_[0][0])
    print("Quality score:", ranker.model.coef_[0][1])
    print("Popularity:", ranker.model.coef_[0][2])


if __name__ == "__main__":
    main()