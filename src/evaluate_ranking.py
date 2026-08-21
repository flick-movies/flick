from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.genre_recommender import score_movies_by_genre
from src.ml_reranker import (
    MovieFeatures,
    calculate_movie_popularity,
    chronological_split,
    load_ranker,
)


@dataclass(frozen=True)
class PairwiseEvaluation:
    heuristic_accuracy: float
    ml_accuracy: float
    users_evaluated: int
    pairs_evaluated: int


def comparison_credit(
    first_score: float,
    second_score: float,
    first_rating: float,
    second_rating: float,
) -> float:
    actual_direction = np.sign(first_rating - second_rating)
    predicted_direction = np.sign(first_score - second_score)

    if predicted_direction == 0:
        return 0.5

    if predicted_direction == actual_direction:
        return 1.0

    return 0.0


def evaluate_pairwise_accuracy(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> PairwiseEvaluation:
    ranker = load_ranker()

    popularity = calculate_movie_popularity(ratings)

    heuristic_user_accuracies: list[float] = []
    ml_user_accuracies: list[float] = []

    total_pairs = 0

    for user_id in ratings["userId"].unique():
        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            continue

        profile, _, test = chronological_split(user_ratings)

        if len(test) < 2:
            continue

        test_movie_ids = test["movieId"].astype(int).tolist()

        scored_movies = score_movies_by_genre(
            user_id=int(user_id),
            user_history=profile,
            reference_ratings=ratings,
            movies=movies,
            movie_ids=test_movie_ids,
        )

        if len(scored_movies) < 2:
            continue

        movie_data = {}

        for movie in scored_movies:
            if movie.movie_id not in popularity:
                continue

            actual_rows = test.loc[
                test["movieId"] == movie.movie_id,
                "rating",
            ]

            if actual_rows.empty:
                continue

            actual_rating = float(actual_rows.iloc[0])

            features = MovieFeatures(
                personal_score=movie.personal_score,
                quality_score=movie.quality_score,
                popularity=popularity[movie.movie_id],
            )

            raw_features = features.as_array().reshape(1, -1)

            scaled_features = ranker.scaler.transform(
                raw_features
            )

            ml_score = float(
                ranker.model.decision_function(
                    scaled_features
                )[0]
            )

            movie_data[movie.movie_id] = (
                actual_rating,
                movie.predicted_rating,
                ml_score,
            )

        movie_ids = list(movie_data.keys())

        heuristic_correct = 0.0
        ml_correct = 0.0
        user_pairs = 0

        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):
                first = movie_data[movie_ids[i]]
                second = movie_data[movie_ids[j]]

                first_rating = first[0]
                second_rating = second[0]

                # Equal ratings tell us nothing about preference order.
                if first_rating == second_rating:
                    continue

                first_heuristic = first[1]
                second_heuristic = second[1]

                first_ml = first[2]
                second_ml = second[2]

                heuristic_correct += comparison_credit(
                    first_score=first_heuristic,
                    second_score=second_heuristic,
                    first_rating=first_rating,
                    second_rating=second_rating,
                )

                ml_correct += comparison_credit(
                    first_score=first_ml,
                    second_score=second_ml,
                    first_rating=first_rating,
                    second_rating=second_rating,
                )

                user_pairs += 1

        if user_pairs == 0:
            continue

        heuristic_user_accuracies.append(
            heuristic_correct / user_pairs
        )

        ml_user_accuracies.append(
            ml_correct / user_pairs
        )

        total_pairs += user_pairs

    if not heuristic_user_accuracies:
        raise ValueError("No valid test pairs were generated")

    return PairwiseEvaluation(
        heuristic_accuracy=float(
            np.mean(heuristic_user_accuracies)
        ),
        ml_accuracy=float( 
            np.mean(ml_user_accuracies)
        ),
        users_evaluated=len(heuristic_user_accuracies),
        pairs_evaluated=total_pairs,
    )