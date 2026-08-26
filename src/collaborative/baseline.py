from __future__ import annotations

import pandas as pd


class MovieAverageBaseline:

    def __init__(self, prior_strength: float = 10.0):
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")

        self.prior_strength = prior_strength

        self.global_mean_: float | None = None
        self.movie_means_: pd.Series | None = None
        self.movie_counts_: pd.Series | None = None
        self.is_fitted_ = False

    def fit(self, ratings: pd.DataFrame) -> MovieAverageBaseline:
        required_columns = {"movieId", "rating"}
        missing = required_columns - set(ratings.columns)

        if missing:
            raise ValueError(
                f"ratings is missing required columns: {sorted(missing)}"
            )

        if ratings.empty:
            raise ValueError("ratings cannot be empty")

        clean_ratings = ratings[["movieId", "rating"]].dropna()

        if clean_ratings.empty:
            raise ValueError("ratings contains no valid rating rows")

        self.global_mean_ = float(clean_ratings["rating"].mean())

        movie_stats = (
            clean_ratings
            .groupby("movieId")["rating"]
            .agg(["mean", "count"])
        )

        self.movie_means_ = movie_stats["mean"]
        self.movie_counts_ = movie_stats["count"]

        self.is_fitted_ = True

        return self

    def predict(
        self,
        user_ids,
        movie_ids,
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError(
                "Model must be fitted before calling predict()"
            )

        if len(user_ids) != len(movie_ids):
            raise ValueError(
                "user_ids and movie_ids must have the same length"
            )

        assert self.global_mean_ is not None
        assert self.movie_means_ is not None
        assert self.movie_counts_ is not None

        predictions = []

        for user_id, movie_id in zip(user_ids, movie_ids):
            if movie_id in self.movie_means_.index:
                movie_mean = float(self.movie_means_.loc[movie_id])
                movie_count = float(self.movie_counts_.loc[movie_id])

                if self.prior_strength > 0:
                    prediction = (
                        movie_count * movie_mean
                        + self.prior_strength * self.global_mean_
                    ) / (
                        movie_count + self.prior_strength
                    )

                    confidence = (
                        movie_count
                        / (movie_count + self.prior_strength)
                    )
                else:
                    prediction = movie_mean
                    confidence = 1.0

            else:
                prediction = self.global_mean_
                confidence = 0.0

            prediction = max(0.0, min(5.0, float(prediction)))

            predictions.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "predicted_score": prediction,
                    "confidence": float(confidence),
                }
            )

        return pd.DataFrame(
            predictions,
            columns=[
                "user_id",
                "movie_id",
                "predicted_score",
                "confidence",
            ],
        )
