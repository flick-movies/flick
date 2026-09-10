import pandas as pd
import pytest
from .evaluator import collaborative_holdout, evaluate_models


def test_chronological_boundaries_and_unknown_movies():
    rows = pd.DataFrame({"userId": [1] * 10 + [2] * 5,
                         "movieId": range(15), "rating": [3.] * 15,
                         "timestamp": [1] * 15}).sample(frac=1, random_state=2)
    train, heldout = collaborative_holdout(rows)
    profile, test = heldout[1]
    assert profile.movieId.tolist() == list(range(6))
    assert test.movieId.tolist() == [8, 9]
    assert set(train.movieId).isdisjoint(test.movieId)
    assert 2 not in heldout
    assert len(train.loc[train.userId == 2]) == 5
    _, predictions = evaluate_models(rows, split="chronological", return_details=True)
    assert predictions.movieId.tolist() == [8, 9]
    assert predictions.user_count.tolist() == [8, 8]
    assert predictions.movie_count.tolist() == [0, 0]


def test_chronological_requires_timestamps():
    with pytest.raises(ValueError, match="timestamp"):
        collaborative_holdout(pd.DataFrame({"userId": [1], "movieId": [1], "rating": [3]}))
