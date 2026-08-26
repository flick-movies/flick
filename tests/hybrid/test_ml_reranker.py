import numpy as np
import pandas as pd
import pytest

from src.hybrid.ml_reranker import (
    MovieFeatures,
    build_pairwise_examples,
    calculate_movie_popularity,
    chronological_split,
)


def test_chronological_split_preserves_time_order():
    ratings = pd.DataFrame(
        {
            "movieId": list(range(10)),
            "rating": [3.0] * 10,
            "timestamp": [50, 10, 90, 20, 80, 30, 70, 40, 100, 60],
        }
    )

    profile, train, test = chronological_split(ratings)

    assert len(profile) == 6
    assert len(train) == 2
    assert len(test) == 2

    assert profile["timestamp"].max() < train["timestamp"].min()
    assert train["timestamp"].max() < test["timestamp"].min()


def test_pairwise_examples_are_balanced():
    movie_features = {
        1: MovieFeatures(
            personal_score=1.0,
            quality_score=0.5,
            popularity=2.0,
        ),
        2: MovieFeatures(
            personal_score=0.0,
            quality_score=0.2,
            popularity=1.0,
        ),
    }

    actual_ratings = {
        1: 5.0,
        2: 2.0,
    }

    X, y = build_pairwise_examples(
        movie_features=movie_features,
        actual_ratings=actual_ratings,
    )

    assert X.shape == (2, 3)
    assert y.tolist() == [1, 0]
    assert np.allclose(X[0], -X[1])


def test_movie_popularity_uses_log_rating_count():
    ratings = pd.DataFrame(
        {
            "movieId": [1, 1, 1, 2],
            "rating": [4.0, 5.0, 3.0, 4.0],
        }
    )

    popularity = calculate_movie_popularity(ratings)

    assert popularity[1] == pytest.approx(np.log1p(3))
    assert popularity[2] == pytest.approx(np.log1p(1))
    assert popularity[1] > popularity[2]