import unittest

from src.content.baselines import (
    calculate_user_baseline,
    calculate_user_baselines,
)
from src.content.schemas import UserRating
from tests.fixtures import TOY_RATINGS


class BaselineTests(unittest.TestCase):
    def test_exact_mean_and_residuals(self) -> None:
        baseline = calculate_user_baseline(
            (UserRating(1, 1, 2.0), UserRating(1, 2, 4.0))
        )

        self.assertEqual(baseline.rating_count, 2)
        self.assertAlmostEqual(baseline.mean_rating, 3.0)
        self.assertEqual(
            [residual.residual for residual in baseline.residuals],
            [-1.0, 1.0],
        )

    def test_high_rating_user_is_centered_on_personal_mean(self) -> None:
        baseline = calculate_user_baseline(
            (
                UserRating(1, 1, 4.0),
                UserRating(1, 2, 4.5),
                UserRating(1, 3, 5.0),
            )
        )

        self.assertAlmostEqual(baseline.mean_rating, 4.5)
        self.assertEqual(
            [residual.residual for residual in baseline.residuals],
            [-0.5, 0.0, 0.5],
        )
        self.assertAlmostEqual(
            sum(residual.residual for residual in baseline.residuals),
            0.0,
        )

    def test_single_rating_user_has_zero_residual(self) -> None:
        baseline = calculate_user_baseline((UserRating(7, 3, 4.0),))

        self.assertEqual(baseline.rating_count, 1)
        self.assertEqual(baseline.mean_rating, 4.0)
        self.assertEqual(baseline.residuals[0].residual, 0.0)

    def test_empty_ratings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_user_baseline(())

    def test_mixed_users_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_user_baseline(
                (UserRating(1, 1, 4.0), UserRating(2, 2, 3.0))
            )

    def test_all_toy_users_receive_baselines(self) -> None:
        baselines = calculate_user_baselines(TOY_RATINGS)

        self.assertEqual(tuple(baselines), (1, 2, 3))
        self.assertTrue(all(item.rating_count == 4 for item in baselines.values()))


if __name__ == "__main__":
    unittest.main()
