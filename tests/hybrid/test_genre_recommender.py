import math

import numpy as np
import pandas as pd
import pytest

from src.hybrid.genre_recommender import (
    calculate_recency_weights,
    calculate_year_weights,
    extract_release_year,
)


def test_extract_release_year():
    assert extract_release_year("The Matrix (1999)") == 1999.0
    assert math.isnan(extract_release_year("Movie Without Year"))


def test_recency_weights_give_newest_rating_full_weight():
    timestamps = pd.array([1_000_000, 2_000_000])

    weights = calculate_recency_weights(
        timestamps=timestamps,
        half_life_years=4.0,
    )

    assert weights[1] == pytest.approx(1.0)
    assert weights[0] < weights[1]


def test_year_weights_penalize_movies_from_different_eras():
    rated_years = np.array([2000.0, 2010.0, np.nan])

    weights = calculate_year_weights(
        rated_years=rated_years,
        candidate_year=2000.0,
        penalty_per_year=0.01,
        minimum_weight=0.5,
    )

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.9)
    assert weights[2] == pytest.approx(1.0)