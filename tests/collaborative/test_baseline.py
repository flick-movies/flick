import pandas as pd
import pytest

from src.collaborative.baseline import MovieAverageBaseline


@pytest.fixture
def ratings():
    return pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 3],
            "movieId": [10, 10, 10, 20, 20, 30],
            "rating": [5.0, 4.0, 3.0, 2.0, 2.0, 5.0],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )


def test_fit_calculates_global_mean(ratings):
    model = MovieAverageBaseline(prior_strength=10)
    model.fit(ratings)

    assert model.global_mean_ == pytest.approx(3.5)


def test_fit_calculates_movie_statistics(ratings):
    model = MovieAverageBaseline(prior_strength=10)
    model.fit(ratings)

    assert model.movie_means_.loc[10] == pytest.approx(4.0)
    assert model.movie_counts_.loc[10] == 3

    assert model.movie_means_.loc[20] == pytest.approx(2.0)
    assert model.movie_counts_.loc[20] == 2


def test_predict_returns_required_columns(ratings):
    model = MovieAverageBaseline(prior_strength=10)
    model.fit(ratings)

    predictions = model.predict(
        user_ids=[1, 2],
        movie_ids=[10, 20],
    )

    assert list(predictions.columns) == [
        "user_id",
        "movie_id",
        "predicted_score",
        "confidence",
    ]

    assert len(predictions) == 2


def test_prediction_is_bayesian_adjusted(ratings):
    model = MovieAverageBaseline(prior_strength=10)
    model.fit(ratings)

    predictions = model.predict(
        user_ids=[1],
        movie_ids=[10],
    )

    # Movie 10:
    # mean = 4.0
    # count = 3
    # global mean = 3.5
    #
    # (3 * 4.0 + 10 * 3.5) / (3 + 10)
    expected = (3 * 4.0 + 10 * 3.5) / 13

    assert predictions.iloc[0]["predicted_score"] == pytest.approx(
        expected
    )


def test_confidence_increases_with_more_ratings(ratings):
    model = MovieAverageBaseline(prior_strength=10)
    model.fit(ratings)

    predictions = model.predict(
        user_ids=[1, 1],
        movie_ids=[10, 20],
    )

    confidence_movie_10 = predictions.iloc[0]["confidence"]
    confidence_movie_20 = predictions.iloc[1]["confidence"]

    assert confidence_movie_10 > confidence_movie_20


def test_unknown_movie_uses_global_average(ratings):
    model = MovieAverageBaseline(prior_strength=10)
    model.fit(ratings)

    predictions = model.predict(
        user_ids=[1],
        movie_ids=[999],
    )

    assert predictions.iloc[0]["predicted_score"] == pytest.approx(
        model.global_mean_
    )

    assert predictions.iloc[0]["confidence"] == 0.0


def test_predict_requires_matching_lengths(ratings):
    model = MovieAverageBaseline()
    model.fit(ratings)

    with pytest.raises(ValueError):
        model.predict(
            user_ids=[1, 2],
            movie_ids=[10],
        )


def test_predict_requires_fit(ratings):
    model = MovieAverageBaseline()

    with pytest.raises(RuntimeError):
        model.predict(
            user_ids=[1],
            movie_ids=[10],
        )