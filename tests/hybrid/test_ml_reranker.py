import numpy as np
import pandas as pd
import pytest

import src.hybrid.ml_reranker as ml_reranker

from src.hybrid.ml_reranker import (
    MovieFeatures,
    build_pairwise_examples,
    build_training_dataset,
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

    assert profile["timestamp"].max() <= train["timestamp"].min()
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_chronological_split_has_no_overlap():
    ratings = pd.DataFrame(
        {
            "movieId": list(range(10)),
            "rating": [3.0] * 10,
            "timestamp": list(range(10)),
        }
    )

    profile, train, test = chronological_split(ratings)

    profile_ids = set(profile["movieId"])
    train_ids = set(train["movieId"])
    test_ids = set(test["movieId"])

    assert profile_ids.isdisjoint(train_ids)
    assert profile_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(test_ids)


def test_chronological_split_covers_every_rating():
    ratings = pd.DataFrame(
        {
            "movieId": list(range(13)),
            "rating": [4.0] * 13,
            "timestamp": list(range(13)),
        }
    )

    profile, train, test = chronological_split(ratings)

    assert len(profile) + len(train) + len(test) == len(ratings)

    combined_ids = set(
        pd.concat(
            [profile["movieId"], train["movieId"], test["movieId"]]
        )
    )

    assert combined_ids == set(ratings["movieId"])


@pytest.mark.parametrize(
    ("profile_fraction", "train_fraction"),
    [
        (-0.1, 0.2),
        (1.1, 0.0),
        (0.6, -0.1),
        (0.6, 1.1),
        (0.8, 0.3),
    ],
)
def test_chronological_split_rejects_invalid_fractions(
    profile_fraction,
    train_fraction,
):
    ratings = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "rating": [3.0, 4.0, 5.0],
            "timestamp": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        chronological_split(
            ratings,
            profile_fraction=profile_fraction,
            train_fraction=train_fraction,
        )


def test_chronological_split_is_deterministic_with_equal_timestamps():
    ratings = pd.DataFrame(
        {
            "movieId": [9, 2, 7, 1, 5, 3, 8, 4, 6, 0],
            "rating": [3.0] * 10,
            "timestamp": [100] * 10,
        }
    )

    first_profile, first_train, first_test = chronological_split(ratings)
    second_profile, second_train, second_test = chronological_split(ratings)

    assert first_profile["movieId"].tolist() == [0, 1, 2, 3, 4, 5]
    assert first_train["movieId"].tolist() == [6, 7]
    assert first_test["movieId"].tolist() == [8, 9]

    assert first_profile.equals(second_profile)
    assert first_train.equals(second_train)
    assert first_test.equals(second_test)

def test_training_dataset_excludes_held_out_test_rows(monkeypatch):
    ratings = pd.DataFrame(
        {
            "userId": ([1] * 10) + ([2] * 10),
            "movieId": (
                list(range(100, 110))
                + list(range(200, 210))
            ),
            "rating": [3.0, 4.0, 5.0, 2.0, 4.5] * 4,
            "timestamp": list(range(10)) + list(range(10)),
        }
    )

    captured_references: dict[int, set[int]] = {}

    def fake_build_user_training_examples(
        user_id,
        profile_ratings,
        pairwise_ratings,
        reference_ratings,
        movies,
    ):
        captured_references[user_id] = set(
            reference_ratings["movieId"].astype(int)
        )

        return (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            np.array([1, 0]),
        )

    monkeypatch.setattr(
        ml_reranker,
        "build_user_training_examples",
        fake_build_user_training_examples,
    )

    build_training_dataset(
        ratings=ratings,
        movies=pd.DataFrame(),
    )

    held_out_movie_ids = {
        108,
        109,
        208,
        209,
    }

    for reference_movie_ids in captured_references.values():
        assert reference_movie_ids.isdisjoint(
            held_out_movie_ids
        )

def test_training_dataset_excludes_target_user_from_reference(
    monkeypatch,
):
    ratings = pd.DataFrame(
        {
            "userId": ([1] * 10) + ([2] * 10),
            "movieId": (
                list(range(100, 110))
                + list(range(200, 210))
            ),
            "rating": [3.0, 4.0, 5.0, 2.0, 4.5] * 4,
            "timestamp": list(range(10)) + list(range(10)),
        }
    )

    captured_reference_users: dict[int, set[int]] = {}

    def fake_build_user_training_examples(
        user_id,
        profile_ratings,
        pairwise_ratings,
        reference_ratings,
        movies,
    ):
        captured_reference_users[user_id] = set(
            reference_ratings["userId"].astype(int)
        )

        return (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
            np.array([1, 0]),
        )

    monkeypatch.setattr(
        ml_reranker,
        "build_user_training_examples",
        fake_build_user_training_examples,
    )

    build_training_dataset(
        ratings=ratings,
        movies=pd.DataFrame(),
    )

    for user_id, reference_users in captured_reference_users.items():
        assert user_id not in reference_users

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