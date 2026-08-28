import unittest

from src.content.errors import UnknownMovieError, UnknownUserError
from src.content.model import ContentModel
from src.content.schemas import MovieMetadata
from tests.fixtures import TOY_MOVIES, TOY_RATINGS


class ContentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ContentModel(TOY_RATINGS, TOY_MOVIES)

    def test_profile_is_built_once_and_cached(self) -> None:
        first = self.model.build_profile(1)
        second = self.model.build_profile(1)

        self.assertIs(first, second)

    def test_batch_prediction_order_is_deterministic(self) -> None:
        results = self.model.predict((2, 1), (8, 9))

        self.assertEqual(
            tuple((result.user_id, result.movie_id) for result in results),
            ((2, 8), (2, 9), (1, 8), (1, 9)),
        )

    def test_batch_predictions_reuse_profiles(self) -> None:
        profile = self.model.build_profile(1)
        self.model.predict((1,), (5, 7, 8, 9, 10))

        self.assertIs(self.model.build_profile(1), profile)

    def test_unknown_user_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnknownUserError, "Unknown user ID: 999"):
            self.model.predict((999,), (8,))

    def test_unknown_movie_is_rejected(self) -> None:
        with self.assertRaisesRegex(UnknownMovieError, "Unknown movie ID: 999"):
            self.model.predict((1,), (999,))

    def test_empty_batch_returns_empty_tuple(self) -> None:
        self.assertEqual(self.model.predict((), (8,)), ())
        self.assertEqual(self.model.predict((1,), ()), ())

    def test_duplicate_movie_ids_are_rejected(self) -> None:
        movies = TOY_MOVIES + (MovieMetadata(1, "Duplicate", ("Drama",)),)

        with self.assertRaisesRegex(ValueError, "Duplicate movie ID: 1"):
            ContentModel(TOY_RATINGS, movies)


if __name__ == "__main__":
    unittest.main()
