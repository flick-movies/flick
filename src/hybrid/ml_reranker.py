from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.hybrid.genre_recommender import score_movies_by_genre

@dataclass(frozen=True)
class HybridCandidate:
    movie_id: int

    # Scores provided by upstream recommendation models
    content_score: float
    collaborative_score: float

    # Supporting features used by the hybrid/reranking layer
    quality_score: float
    popularity: float

@dataclass
class TrainedRanker:
    scaler: StandardScaler
    model: LogisticRegression


@dataclass(frozen=True)
class MovieFeatures:
    personal_score: float
    quality_score: float
    popularity: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.personal_score,
                self.quality_score,
                self.popularity,
            ],
            dtype=float,
        )
    
def calculate_movie_popularity(
    ratings: pd.DataFrame,
) -> dict[int, float]:
    rating_counts = ratings.groupby("movieId").size() # takes number of ratings for a movie

    return {
        int(movie_id): float(np.log1p(count)) # ln(1 + no. of ratings)
        for movie_id, count in rating_counts.items()
    }

def create_pairwise_example(
    first: MovieFeatures,
    second: MovieFeatures,
    first_rating: float,
    second_rating: float,
) -> tuple[np.ndarray, int]:
    if first_rating == second_rating:
        raise ValueError("Pair must contain different ratings")

    difference = first.as_array() - second.as_array()

    label = 1 if first_rating > second_rating else 0 # categorizing success vs failure

    return difference, label

def chronological_split(
    user_ratings: pd.DataFrame,
    profile_fraction: float = 0.60,
    train_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= profile_fraction <= 1.0:
        raise ValueError("profile_fraction must be between 0 and 1")

    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if profile_fraction + train_fraction > 1.0:
        raise ValueError(
            "profile_fraction + train_fraction cannot exceed 1"
        )

    ordered = (
        user_ratings
        .sort_values(["timestamp", "movieId"], kind="stable")
        .reset_index(drop=True)
    )

    total = len(ordered)

    profile_end = int(total * profile_fraction)
    train_end = int(total * (profile_fraction + train_fraction))

    profile = ordered.iloc[:profile_end].copy()
    train = ordered.iloc[profile_end:train_end].copy()
    test = ordered.iloc[train_end:].copy()

    return profile, train, test

def build_pairwise_examples(
    movie_features: dict[int, MovieFeatures],
    actual_ratings: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []

    movie_ids = list(movie_features.keys())

    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            first_id = movie_ids[i]
            second_id = movie_ids[j]

            first_rating = actual_ratings[first_id]
            second_rating = actual_ratings[second_id]

            if first_rating == second_rating:
                continue

            first_features = movie_features[first_id].as_array()
            second_features = movie_features[second_id].as_array()

            if first_rating > second_rating:
                preferred = first_features
                less_preferred = second_features
            else:
                preferred = second_features
                less_preferred = first_features

            difference = preferred - less_preferred

            feature_rows.append(difference)
            labels.append(1)

            feature_rows.append(-difference)
            labels.append(0)

    return (
        np.vstack(feature_rows),
        np.array(labels, dtype=int),
    )

def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
) -> TrainedRanker:
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    return TrainedRanker(
        scaler=scaler,
        model=model,
    )

def build_user_training_examples(
    user_id: int,
    profile_ratings: pd.DataFrame,
    pairwise_ratings: pd.DataFrame,
    reference_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(pairwise_ratings) < 2:
        return None

    train_movie_ids = (
        pairwise_ratings["movieId"]
        .astype(int)
        .tolist()
    )

    popularity = calculate_movie_popularity(
        reference_ratings
    )

    scored_movies = score_movies_by_genre(
        user_id=user_id,
        user_history=profile_ratings,
        reference_ratings=reference_ratings,
        movies=movies,
        movie_ids=train_movie_ids,
    )

    movie_features: dict[int, MovieFeatures] = {}
    actual_ratings: dict[int, float] = {}

    for movie in scored_movies:
        movie_id = movie.movie_id

        if movie_id not in popularity:
            continue

        rating_rows = pairwise_ratings.loc[
            pairwise_ratings["movieId"] == movie_id,
            "rating",
        ]

        if rating_rows.empty:
            continue

        movie_features[movie_id] = MovieFeatures(
            personal_score=movie.personal_score,
            quality_score=movie.quality_score,
            popularity=popularity[movie_id],
        )

        actual_ratings[movie_id] = float(
            rating_rows.iloc[0]
        )

    if len(movie_features) < 2:
        return None

    if len(set(actual_ratings.values())) < 2:
        return None

    return build_pairwise_examples(
        movie_features=movie_features,
        actual_ratings=actual_ratings,
    )


def build_training_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    user_splits: dict[
        int,
        tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    reference_parts: list[pd.DataFrame] = []

    for raw_user_id in ratings["userId"].unique():
        user_id = int(raw_user_id)

        user_ratings = ratings.loc[
            ratings["userId"] == user_id
        ].copy()

        if len(user_ratings) < 10:
            # These users are not used for pairwise ML training,
            # so they have no held-out test partition here.
            reference_parts.append(user_ratings)
            continue

        profile, train, _ = chronological_split(
            user_ratings
        )

        user_splits[user_id] = (
            profile,
            train,
        )

        reference_parts.append(profile)
        reference_parts.append(train)

    if not reference_parts:
        raise ValueError(
            "No leakage-safe reference ratings were generated"
        )

    global_reference_ratings = pd.concat(
        reference_parts,
        ignore_index=True,
    )

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    users_used = 0

    for user_id, (profile, train) in user_splits.items():
        # Do not let this user's own pairwise-training ratings
        # influence population-level quality/popularity features.
        user_reference_ratings = (
            global_reference_ratings.loc[
                global_reference_ratings["userId"]
                != user_id
            ]
        )

        result = build_user_training_examples(
            user_id=user_id,
            profile_ratings=profile,
            pairwise_ratings=train,
            reference_ratings=user_reference_ratings,
            movies=movies,
        )

        if result is None:
            continue

        X_user, y_user = result

        all_features.append(X_user)
        all_labels.append(y_user)

        users_used += 1

    if not all_features:
        raise ValueError(
            "No training examples were generated"
        )

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)

    return X, y, users_used

def score_with_ranker(
    features: MovieFeatures,
    ranker: TrainedRanker,
) -> float:
    X = features.as_array().reshape(1, -1)

    X_scaled = ranker.scaler.transform(X)

    score = ranker.model.decision_function(X_scaled)[0]

    return float(score)

def rerank_candidates_with_ml(
    candidates,
    popularity: dict[int, float],
    ranker: TrainedRanker,
):
    scored = []

    for movie in candidates:
        features = MovieFeatures(
            personal_score=movie.personal_score,
            quality_score=movie.quality_score,
            popularity=popularity[movie.movie_id],
        )

        ml_score = score_with_ranker(
            features=features,
            ranker=ranker,
        )

        scored.append((movie, ml_score))

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored

def save_ranker(
    ranker: TrainedRanker,
    path: str = "src/models/ml_reranker.joblib"
) -> None:
    joblib.dump(ranker, path)


def load_ranker(
    path: str = "src/models/ml_reranker.joblib"
) -> TrainedRanker:
    return joblib.load(path)

@dataclass(frozen=True)
class MLRerankedRecommendation:
    movie_id: int
    title: str
    genres: str
    ml_score: float
    personal_score: float
    quality_score: float
    popularity: float


def recommend_with_ml(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    limit: int = 5,
) -> list[MLRerankedRecommendation]:
    ranker = load_ranker()

    user_history = ratings.loc[
        ratings["userId"] == user_id
    ].copy()

    if user_history.empty:
        raise ValueError(f"User {user_id} has no ratings")

    watched_ids = set(
        user_history["movieId"].astype(int)
    )

    unseen_movie_ids = (
        movies.loc[
            ~movies["movieId"].isin(watched_ids),
            "movieId",
        ]
        .astype(int)
        .tolist()
    )

    scored_movies = score_movies_by_genre(
        user_id=user_id,
        user_history=user_history,
        reference_ratings=ratings,
        movies=movies,
        movie_ids=unseen_movie_ids,
    )

    popularity = calculate_movie_popularity(ratings)

    recommendations: list[MLRerankedRecommendation] = []

    for movie in scored_movies:
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

        recommendations.append(
            MLRerankedRecommendation(
                movie_id=movie.movie_id,
                title=movie.title,
                genres=movie.genres,
                ml_score=ml_score,
                personal_score=movie.personal_score,
                quality_score=movie.quality_score,
                popularity=popularity[movie.movie_id],
            )
        )

    recommendations.sort(
        key=lambda recommendation: recommendation.ml_score,
        reverse=True,
    )

    return recommendations[:limit]