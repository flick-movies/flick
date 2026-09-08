from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..baseline import MovieAverageBaseline
from ..matrix_factorization import BiasedMatrixFactorization

from .metrics import mae, rmse, accuracy_within


def evaluate_models(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    return_details: bool = False,
):
    required_columns = {"userId", "movieId", "rating"}

    missing = required_columns - set(ratings.columns)

    if missing:
        raise ValueError(
            f"ratings is missing required columns: {sorted(missing)}"
        )

    clean_ratings = ratings[
        ["userId", "movieId", "rating"]
    ].dropna()

    if clean_ratings.empty:
        raise ValueError("ratings cannot be empty")

    train, test = train_test_split(
        clean_ratings,
        test_size=test_size,
        random_state=random_state,
    )

    # Train both models
    baseline = MovieAverageBaseline()
    baseline.fit(train)

    matrix_factorization = BiasedMatrixFactorization(
        random_state=random_state
    )
    matrix_factorization.fit(train)

    # Make predictions
    user_ids = test["userId"].tolist()
    movie_ids = test["movieId"].tolist()
    actual = test["rating"].to_numpy()

    baseline_predictions = baseline.predict(
        user_ids,
        movie_ids,
    )

    mf_predictions = matrix_factorization.predict(
        user_ids,
        movie_ids,
    )

    baseline_scores = baseline_predictions["predicted_score"].to_numpy()
    mf_scores = mf_predictions["predicted_score"].to_numpy()

    # Calculate metrics
    results = {
        "Baseline": {
            "RMSE": rmse(actual, baseline_scores),
            "MAE": mae(actual, baseline_scores),
            "Accuracy within 0.5": accuracy_within(
                actual,
                baseline_scores,
                tolerance=0.5,
            ),
        },
        "Matrix Factorization": {
            "RMSE": rmse(actual, mf_scores),
            "MAE": mae(actual, mf_scores),
            "Accuracy within 0.5": accuracy_within(
                actual,
                mf_scores,
                tolerance=0.5,
            ),
        },
    }

    # Evaluate the old inference rule using identical trained weights.
    matrix_factorization.shrink_latent = False
    original = matrix_factorization.predict(user_ids, movie_ids)
    results["MF without shrinkage"] = summarize(actual, original["predicted_score"])
    predictions = test.reset_index(drop=True).copy()
    predictions["baseline_score"] = baseline_scores
    predictions["mf_score"] = mf_scores
    predictions["original_mf_score"] = original["predicted_score"].to_numpy()
    predictions["confidence"] = mf_predictions["confidence"].to_numpy()
    predictions["absolute_error"] = abs(predictions["rating"] - predictions["mf_score"])
    predictions["user_count"] = predictions["userId"].map(train["userId"].value_counts()).fillna(0).astype(int)
    predictions["movie_count"] = predictions["movieId"].map(train["movieId"].value_counts()).fillna(0).astype(int)
    if return_details:
        return results, predictions
    return results

def summarize(actual, predicted):
    return {"RMSE": rmse(actual, predicted), "MAE": mae(actual, predicted),
            "Accuracy within 0.5": accuracy_within(actual, predicted)}


def confidence_report(predictions):
    """Nonoverlapping buckets: [lower, upper), with 1 included in the last."""
    rows = []
    for lower, upper in zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1]):
        confidence = predictions["confidence"]
        mask = (confidence >= lower) & ((confidence <= upper) if upper == 1 else (confidence < upper))
        subset = predictions.loc[mask]
        values = summarize(subset["rating"], subset["mf_score"]) if len(subset) else {
            "RMSE": float("nan"), "MAE": float("nan"), "Accuracy within 0.5": float("nan")}
        rows.append({"Confidence": f"[{lower:.2f}, {upper:.2f}{']' if upper == 1 else ')'}",
                     "Count": len(subset), **values})
    return pd.DataFrame(rows)


def sparse_report(predictions):
    """Overlapping diagnostic groups; counts refer only to training ratings."""
    rows = []
    for name, mask in {
        "User count <= 5": predictions["user_count"] <= 5,
        "Movie count <= 5": predictions["movie_count"] <= 5,
        "Unknown user": predictions["user_count"] == 0,
        "Unknown movie": predictions["movie_count"] == 0,
    }.items():
        subset = predictions.loc[mask]
        for model, column in [("MF", "mf_score"), ("MF without shrinkage", "original_mf_score")]:
            values = summarize(subset["rating"], subset[column]) if len(subset) else {
                "RMSE": float("nan"), "MAE": float("nan"), "Accuracy within 0.5": float("nan")}
            rows.append({"Group": name, "Model": model, "Count": len(subset), **values})
    return pd.DataFrame(rows)
