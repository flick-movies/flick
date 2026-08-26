import unittest

from src.content.baselines import calculate_user_baseline
from src.content.genres import (
    aggregate_genre_preferences,
    normalized_genre_contributions,
)
from src.content.schemas import MovieMetadata, UserRating
from src.data_processing.movielens import (
    movie_metadata_from_record,
    parse_genres,
)
from tests.fixtures import TOY_MOVIES_BY_ID, TOY_RATINGS


class GenreTests(unittest.TestCase):
    def test_single_genre_receives_full_residual(self) -> None:
        contributions = normalized_genre_contributions(1.0, ("Drama",))

        self.assertEqual(contributions, {"Drama": 1.0})

    def test_five_genres_share_residual(self) -> None:
        genres = ("Action", "Adventure", "Comedy", "Drama", "Sci-Fi")
        contributions = normalized_genre_contributions(1.0, genres)

        self.assertEqual(len(contributions), 5)
        self.assertTrue(all(value == 0.2 for value in contributions.values()))
        self.assertAlmostEqual(sum(contributions.values()), 1.0)

    def test_duplicate_genres_do_not_inflate_signal(self) -> None:
        contributions = normalized_genre_contributions(
            1.0,
            ("Drama", "Drama", "drama", "Comedy"),
        )

        self.assertEqual(contributions, {"Drama": 0.5, "Comedy": 0.5})
        self.assertAlmostEqual(sum(contributions.values()), 1.0)

    def test_missing_genres_produce_no_signal(self) -> None:
        self.assertEqual(normalized_genre_contributions(1.0, ()), {})
        self.assertEqual(parse_genres(None), ())
        self.assertEqual(parse_genres("(no genres listed)"), ())

    def test_movielens_genres_are_cleaned_and_deduplicated(self) -> None:
        genres = parse_genres(" Action |Comedy|Action||comedy ")

        self.assertEqual(genres, ("Action", "Comedy"))

    def test_movie_record_is_converted_to_typed_metadata(self) -> None:
        movie = movie_metadata_from_record(
            {
                "movieId": 42,
                "title": "Example Movie (1999)",
                "genres": "Drama|Thriller",
            }
        )

        self.assertEqual(movie.movie_id, 42)
        self.assertEqual(movie.genres, ("Drama", "Thriller"))
        self.assertEqual(movie.release_year, 1999)

    def test_raw_genre_preferences_are_aggregated(self) -> None:
        user_ratings = tuple(
            rating for rating in TOY_RATINGS if rating.user_id == 1
        )
        baseline = calculate_user_baseline(user_ratings)
        preferences = aggregate_genre_preferences(
            baseline,
            TOY_MOVIES_BY_ID,
        )
        by_genre = {preference.genre: preference for preference in preferences}

        self.assertGreater(by_genre["Sci-Fi"].total_contribution, 0.0)
        self.assertLess(by_genre["Romance"].total_contribution, 0.0)
        self.assertEqual(by_genre["Sci-Fi"].movie_count, 2)

    def test_unknown_movie_metadata_is_skipped(self) -> None:
        baseline = calculate_user_baseline(
            (UserRating(1, 999, 4.0), UserRating(1, 1, 2.0))
        )
        movies = {1: MovieMetadata(1, "Known", ("Drama",))}

        preferences = aggregate_genre_preferences(baseline, movies)

        self.assertEqual(len(preferences), 1)
        self.assertEqual(preferences[0].genre, "Drama")


if __name__ == "__main__":
    unittest.main()
