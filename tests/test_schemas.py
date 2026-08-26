import json
import unittest

from src.content.schemas import (
    MovieMetadata,
    PredictionResult,
    ReasonSignal,
    UserRating,
)
from tests.fixtures import TOY_MOVIES, TOY_RATINGS


class SchemaTests(unittest.TestCase):
    def test_toy_objects_load(self) -> None:
        self.assertEqual(len(TOY_MOVIES), 10)
        self.assertEqual(len({rating.user_id for rating in TOY_RATINGS}), 3)

    def test_prediction_result_serializes(self) -> None:
        result = PredictionResult(
            user_id=1,
            movie_id=8,
            predicted_score=4.6,
            confidence=0.75,
            reason_signals=(
                ReasonSignal("genre", "Sci-Fi", 0.8, 2.0),
            ),
        )

        serialized = json.dumps(result.to_dict())

        self.assertIn('"predicted_score": 4.6', serialized)
        self.assertIn('"feature_value": "Sci-Fi"', serialized)

    def test_rating_below_zero_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UserRating(1, 1, -0.5)

    def test_rating_above_five_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UserRating(1, 1, 5.5)

    def test_prediction_score_outside_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PredictionResult(1, 1, 5.1, 0.5)

    def test_movie_title_is_required(self) -> None:
        with self.assertRaises(ValueError):
            MovieMetadata(1, "")


if __name__ == "__main__":
    unittest.main()
