from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from src.collaborative.matrix_factorization import (
    BiasedMatrixFactorization,
)


def main() -> None:

    project_root = Path(__file__).resolve().parents[3]

    ratings_path = project_root / "data" / "ratings.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {ratings_path}"
        )
    # Load ratings.
    ratings = pd.read_csv(ratings_path)

    print("=" * 70)
    print("MATRIX FACTORIZATION CONFIDENCE TEST")
    print("=" * 70)

    print(f"Ratings loaded: {len(ratings):,}")
    print()

    # Train the collaborative model.
    print("Training matrix factorization...")

    model = BiasedMatrixFactorization(
        n_factors=20,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=20,
        prior_strength=5.0,
        random_state=42,
    )

    model.fit(ratings)

    print("Training complete.")
    print()
    

    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    low_rating_user = int(user_counts.idxmin())
    high_rating_user = int(user_counts.idxmax())

    low_rating_movie = int(movie_counts.idxmin())
    high_rating_movie = int(movie_counts.idxmax())

    print("Selected examples:")
    print()

    print(
        f"Low-history user:   {low_rating_user} "
        f"({user_counts[low_rating_user]} ratings)"
    )

    print(
        f"High-history user:  {high_rating_user} "
        f"({user_counts[high_rating_user]} ratings)"
    )

    print(
        f"Low-history movie:  {low_rating_movie} "
        f"({movie_counts[low_rating_movie]} ratings)"
    )

    print(
        f"High-history movie: {high_rating_movie} "
        f"({movie_counts[high_rating_movie]} ratings)"
    )

    print()

    test_cases = [
        (
            "Low user + low movie",
            low_rating_user,
            low_rating_movie,
        ),
        (
            "Low user + high movie",
            low_rating_user,
            high_rating_movie,
        ),
        (
            "High user + low movie",
            high_rating_user,
            low_rating_movie,
        ),
        (
            "High user + high movie",
            high_rating_user,
            high_rating_movie,
        ),
    ]

    print("=" * 70)
    print("CONFIDENCE RESULTS")
    print("=" * 70)

    predictions = []

    for name, user_id, movie_id in test_cases:
        result = model.predict(
            user_ids=[user_id],
            movie_ids=[movie_id],
        )

        row = result.iloc[0]

        predicted_score = float(
            row["predicted_score"]
        )

        confidence = float(
            row["confidence"]
        )

        predictions.append(
            {
                "case": name,
                "user_id": user_id,
                "movie_id": movie_id,
                "user_ratings": int(user_counts[user_id]),
                "movie_ratings": int(movie_counts[movie_id]),
                "predicted_score": predicted_score,
                "confidence": confidence,
            }
        )

    results = pd.DataFrame(predictions)

    print(
        results.to_string(
            index=False,
            formatters={
                "predicted_score": "{:.3f}".format,
                "confidence": "{:.3f}".format,
            },
        )
    )

    print()

    # Test the confidence calculation directly.
    print("=" * 70)
    print("EXPECTED CONFIDENCE CHECK")
    print("=" * 70)

    prior = model.prior_strength

    print(f"Prior strength: {prior}")
    print()

    for name, user_id, movie_id in test_cases:
        user_count = int(user_counts[user_id])
        movie_count = int(movie_counts[movie_id])

        expected_user_confidence = (
            user_count
            / (user_count + prior)
        )

        expected_movie_confidence = (
            movie_count
            / (movie_count + prior)
        )

        expected_confidence = (
            expected_user_confidence
            * expected_movie_confidence
        )

        print(name)
        print(
            f"  User confidence:  "
            f"{expected_user_confidence:.4f}"
        )
        print(
            f"  Movie confidence: "
            f"{expected_movie_confidence:.4f}"
        )
        print(
            f"  Final confidence: "
            f"{expected_confidence:.4f}"
        )
        print()

    # Unknown user/movie tests.
    print("=" * 70)
    print("UNKNOWN USER / MOVIE TEST")
    print("=" * 70)

    unknown_user = 999999999
    unknown_movie = 999999999

    unknown_cases = [
        (
            "Known user + known movie",
            high_rating_user,
            high_rating_movie,
        ),
        (
            "Unknown user + known movie",
            unknown_user,
            high_rating_movie,
        ),
        (
            "Known user + unknown movie",
            high_rating_user,
            unknown_movie,
        ),
        (
            "Unknown user + unknown movie",
            unknown_user,
            unknown_movie,
        ),
    ]

    unknown_results = []

    for name, user_id, movie_id in unknown_cases:
        result = model.predict(
            user_ids=[user_id],
            movie_ids=[movie_id],
        )

        row = result.iloc[0]

        unknown_results.append(
            {
                "case": name,
                "predicted_score": float(
                    row["predicted_score"]
                ),
                "confidence": float(
                    row["confidence"]
                ),
            }
        )

    print(
        pd.DataFrame(unknown_results).to_string(
            index=False,
            formatters={
                "predicted_score": "{:.3f}".format,
                "confidence": "{:.3f}".format,
            },
        )
    )

    print()
    print("=" * 70)
    print("CONFIDENCE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
