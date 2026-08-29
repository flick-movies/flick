# src/evaluate_ranking.py
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.collaborative.baseline import MovieAverageBaseline
from src.hybrid.genre_recommender import score_movies_by_genre
from src.hybrid.ml_reranker import (
    MovieFeatures,
    calculate_movie_popularity,
    chronological_split,
    load_ranker,
)


@dataclass(frozen=True)
class PairwiseEvaluation:
    baseline_accuracy: float
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

    train_ratings_list = []
    user_test_data = {}

    for user_id in ratings["userId"].unique():
        user_ratings = ratings.loc[ratings["userId"] == user_id].copy()

        if len(user_ratings) < 10:
            train_ratings_list.append(user_ratings)
            continue

        profile, train, test = chronological_split(user_ratings)

        train_ratings_list.append(profile)
        train_ratings_list.append(train)

        if len(test) >= 2:
            user_test_data[int(user_id)] = (profile, test)

    all_train_ratings = pd.concat(train_ratings_list, ignore_index=True)

    baseline_model = MovieAverageBaseline(prior_strength=10.0)
    baseline_model.fit(all_train_ratings)
    baseline_user_accuracies: list[float] = []
    heuristic_user_accuracies: list[float] = []
    ml_user_accuracies: list[float] = []

    total_pairs = 0

    for user_id, (profile, test) in user_test_data.items():
        test_movie_ids = test["movieId"].astype(int).tolist()

        reference_ratings = ratings.loc[
            ratings["userId"] != user_id
        ]

        popularity = calculate_movie_popularity(
            reference_ratings
        )

        scored_movies = score_movies_by_genre(
            user_id=user_id,
            user_history=profile,
            reference_ratings=reference_ratings,
            movies=movies,
            movie_ids=test_movie_ids,
        )

        if len(scored_movies) < 2:
            continue
        baseline_preds = baseline_model.predict(
            user_ids=[user_id] * len(test_movie_ids),
            movie_ids=test_movie_ids,
        ).set_index("movie_id")["predicted_score"].to_dict()

        movie_data = {}

        for movie in scored_movies:
            if movie.movie_id not in popularity or movie.movie_id not in baseline_preds:
                continue

            actual_rows = test.loc[test["movieId"] == movie.movie_id, "rating"]
            if actual_rows.empty:
                continue

            actual_rating = float(actual_rows.iloc[0])
            features = MovieFeatures(
                personal_score=movie.personal_score,
                quality_score=movie.quality_score,
                popularity=popularity[movie.movie_id],
            )
            raw_features = features.as_array().reshape(1, -1)
            scaled_features = ranker.scaler.transform(raw_features)
            ml_score = float(
                ranker.model.decision_function(scaled_features)[0])

            baseline_score = float(baseline_preds[movie.movie_id])

            movie_data[movie.movie_id] = {
                "actual": actual_rating,
                "heuristic": movie.predicted_rating,
                "ml": ml_score,
                "baseline": baseline_score,
            }

        movie_ids = list(movie_data.keys())
        if len(movie_ids) < 2:
            continue

        baseline_correct = 0.0
        heuristic_correct = 0.0
        ml_correct = 0.0
        user_pairs = 0

        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):
                first = movie_data[movie_ids[i]]
                second = movie_data[movie_ids[j]]

                if first["actual"] == second["actual"]:
                    continue

                baseline_correct += comparison_credit(
                    first_score=first["baseline"],
                    second_score=second["baseline"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                heuristic_correct += comparison_credit(
                    first_score=first["heuristic"],
                    second_score=second["heuristic"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                ml_correct += comparison_credit(
                    first_score=first["ml"],
                    second_score=second["ml"],
                    first_rating=first["actual"],
                    second_rating=second["actual"],
                )

                user_pairs += 1

        if user_pairs == 0:
            continue

        baseline_user_accuracies.append(baseline_correct / user_pairs)
        heuristic_user_accuracies.append(heuristic_correct / user_pairs)
        ml_user_accuracies.append(ml_correct / user_pairs)
        total_pairs += user_pairs

    if not baseline_user_accuracies:
        raise ValueError("No valid test pairs were generated")

    return PairwiseEvaluation(
        baseline_accuracy=float(np.mean(baseline_user_accuracies)),
        heuristic_accuracy=float(np.mean(heuristic_user_accuracies)),
        ml_accuracy=float(np.mean(ml_user_accuracies)),
        users_evaluated=len(baseline_user_accuracies),
        pairs_evaluated=total_pairs,
    )
