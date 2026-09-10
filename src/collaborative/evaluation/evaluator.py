from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..baseline import MovieAverageBaseline
from ..matrix_factorization import BiasedMatrixFactorization

from .metrics import mae, rmse, accuracy_within



def collaborative_holdout(ratings: pd.DataFrame):
    """Per-user chronological 60/20/20; fit profile + train, hold out test."""
    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
    if ratings[list(required)].isna().any().any():
        raise ValueError("Chronological evaluation requires nonmissing values")
    training = []
    heldout = {}
    for uid, history in ratings.groupby("userId", sort=False):
        if len(history) < 10:
            training.append(history)
            continue
        ordered = history.sort_values(["timestamp", "movieId"], kind="stable")
        profile_end = int(len(ordered) * .6)
        train_end = int(len(ordered) * .8)
        profile = ordered.iloc[:profile_end].copy()
        train = ordered.iloc[profile_end:train_end].copy()
        test = ordered.iloc[train_end:].copy()
        training.extend([profile, train])
        if len(test) >= 2:
            heldout[int(uid)] = (profile, test)
    if not training:
        raise ValueError("No training ratings")
    return pd.concat(training, ignore_index=True), heldout


def evaluate_models(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    return_details: bool = False,
    split: str = "random",
):
    if split == "chronological":
        if test_size != 0.2:
            raise ValueError("Team chronological evaluation fixes the held-out fraction at 0.2")
        train, user_test_data = collaborative_holdout(ratings)
        if not user_test_data:
            raise ValueError("No eligible chronological held-out users")
        test = pd.concat([test for _, test in user_test_data.values()], ignore_index=True)
    elif split == "random":
        required = {"userId", "movieId", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
        clean_ratings = ratings[["userId", "movieId", "rating"]].dropna()
        train, test = train_test_split(clean_ratings, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("split must be 'chronological' or 'random'")

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
