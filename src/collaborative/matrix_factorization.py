from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelWeights:
    global_mean: float
    user_biases: np.ndarray
    movie_biases: np.ndarray
    user_factors: np.ndarray
    movie_factors: np.ndarray
    user_to_idx: dict[int, int]
    movie_to_idx: dict[int, int]
    user_counts: dict[int, int]
    movie_counts: dict[int, int]


class BiasedMatrixFactorization:
    # Collaborative Filtering using Latent Taste Vectors & Biases (FunkSVD).

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        n_epochs: int = 20,
        prior_strength: float = 5.0,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.prior_strength = prior_strength
        self.random_state = random_state

        self.weights_: ModelWeights | None = None
        self.is_fitted_ = False

    def fit(self, ratings: pd.DataFrame) -> BiasedMatrixFactorization:
        required_cols = {"userId", "movieId", "rating"}
        missing = required_cols - set(ratings.columns)
        if missing:
            raise ValueError(
                f"Ratings DataFrame is missing columns: {sorted(missing)}")

        clean_ratings = ratings[["userId", "movieId", "rating"]].dropna()
        if clean_ratings.empty:
            raise ValueError("Ratings cannot be empty.")

        rng = np.random.RandomState(self.random_state)

        unique_users = clean_ratings["userId"].unique()
        unique_movies = clean_ratings["movieId"].unique()

        user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        movie_to_idx = {int(mid): idx for idx, mid in enumerate(unique_movies)}

        n_users = len(user_to_idx)
        n_movies = len(movie_to_idx)
        user_counts = clean_ratings["userId"].value_counts().to_dict()
        movie_counts = clean_ratings["movieId"].value_counts().to_dict()
        user_indices = clean_ratings["userId"].map(
            user_to_idx).to_numpy(dtype=np.int32)
        movie_indices = clean_ratings["movieId"].map(
            movie_to_idx).to_numpy(dtype=np.int32)
        rating_values = clean_ratings["rating"].to_numpy(dtype=np.float64)

        global_mean = float(rating_values.mean())
        user_biases = np.zeros(n_users, dtype=np.float64)
        movie_biases = np.zeros(n_movies, dtype=np.float64)

        user_factors = rng.normal(0.0, 0.05, size=(n_users, self.n_factors))
        movie_factors = rng.normal(0.0, 0.05, size=(n_movies, self.n_factors))

        lr = self.learning_rate
        reg = self.regularization
        n_samples = len(rating_values)

        for epoch in range(self.n_epochs):
            shuffle_order = rng.permutation(n_samples)

            for sample_idx in shuffle_order:
                u = user_indices[sample_idx]
                i = movie_indices[sample_idx]
                r = rating_values[sample_idx]

                # Current prediction
                pred = (
                    global_mean
                    + user_biases[u]
                    + movie_biases[i]
                    + np.dot(user_factors[u], movie_factors[i])
                )
                err = r - pred

                user_biases[u] += lr * (err - reg * user_biases[u])
                movie_biases[i] += lr * (err - reg * movie_biases[i])

                u_factors_prev = user_factors[u].copy()
                user_factors[u] += lr * \
                    (err * movie_factors[i] - reg * user_factors[u])
                movie_factors[i] += lr * \
                    (err * u_factors_prev - reg * movie_factors[i])

        self.weights_ = ModelWeights(
            global_mean=global_mean,
            user_biases=user_biases,
            movie_biases=movie_biases,
            user_factors=user_factors,
            movie_factors=movie_factors,
            user_to_idx=user_to_idx,
            movie_to_idx=movie_to_idx,
            user_counts={int(k): int(v) for k, v in user_counts.items()},
            movie_counts={int(k): int(v) for k, v in movie_counts.items()},
        )
        self.is_fitted_ = True
        return self

    def predict(
        self,
        user_ids: list[int] | pd.Series,
        movie_ids: list[int] | pd.Series,
    ) -> pd.DataFrame:

        if not self.is_fitted_ or self.weights_ is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        if len(user_ids) != len(movie_ids):
            raise ValueError(
                "user_ids and movie_ids must have the same length")

        w = self.weights_
        predictions = []

        for uid, mid in zip(user_ids, movie_ids):
            uid = int(uid)
            mid = int(mid)

            has_user = uid in w.user_to_idx
            has_movie = mid in w.movie_to_idx

            if has_user and has_movie:
                u_idx = w.user_to_idx[uid]
                m_idx = w.movie_to_idx[mid]

                score = (
                    w.global_mean
                    + w.user_biases[u_idx]
                    + w.movie_biases[m_idx]
                    + np.dot(w.user_factors[u_idx], w.movie_factors[m_idx])
                )

                u_count = w.user_counts.get(uid, 0)
                m_count = w.movie_counts.get(mid, 0)
                u_conf = u_count / (u_count + self.prior_strength)
                m_conf = m_count / (m_count + self.prior_strength)
                confidence = u_conf * m_conf

            elif has_user:

                u_idx = w.user_to_idx[uid]
                score = w.global_mean + w.user_biases[u_idx]
                confidence = 0.1

            elif has_movie:

                m_idx = w.movie_to_idx[mid]
                score = w.global_mean + w.movie_biases[m_idx]
                m_count = w.movie_counts.get(mid, 0)
                confidence = 0.5 * (m_count / (m_count + self.prior_strength))

            else:

                score = w.global_mean
                confidence = 0.0

            clamped_score = max(0.5, min(5.0, float(score)))

            predictions.append(
                {
                    "user_id": uid,
                    "movie_id": mid,
                    "predicted_score": clamped_score,
                    "confidence": float(confidence),
                }
            )

        return pd.DataFrame(
            predictions,
            columns=["user_id", "movie_id", "predicted_score", "confidence"],
        )

    def save(self, filepath: str | Path = "src/models/collaborative_mf.joblib") -> None:
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path = "src/models/collaborative_mf.joblib") -> BiasedMatrixFactorization:
        return joblib.load(filepath)
