import unittest

from src.content import (
    ContentModel, MovieMetadata, PredictionResult, ProfileConfig, ScoringConfig,
    UserRating, build_profile, predict_batch, predict_one,
)


DAY = 86400
MOVIES = (
    MovieMetadata(1, "Old favorite", ("Drama",)),
    MovieMetadata(2, "Recent dislike", ("drama", " DRAMA ")),
    MovieMetadata(3, "Other", ("Comedy",)),
)
CANDIDATE = MovieMetadata(100, "Unseen, outside training catalog", ("Drama",))


class ReliabilityTests(unittest.TestCase):
    def test_recency_weighted_shrinkage_matches_hand_calculation(self):
        ratings = (UserRating(1, 1, 5, 0), UserRating(1, 2, 1, 10 * DAY))
        profile = build_profile(
            1, ratings, MOVIES, ProfileConfig(recency_half_life_days=10),
        )
        # Baseline 3; old weight .75; new weight 1; total .75*2 - 2 = -.5.
        self.assertEqual(profile.baseline, 3)
        self.assertEqual(profile.evidence_for(" DRAMA "), 2)
        self.assertEqual(len(profile.genre_preferences), 1)
        self.assertAlmostEqual(profile.effective_evidence_for("Drama"), 1.75)
        self.assertAlmostEqual(profile.genre_preferences[0].mean_contribution, -.5 / 1.75)
        self.assertAlmostEqual(profile.preference_for("Drama"), -.5 / 6.75)
        result = predict_one(profile, CANDIDATE, include_debug=True)
        self.assertAlmostEqual(result.predicted_score, 3 - .5 / 6.75)
        self.assertAlmostEqual(result.confidence, (2 / 7) * (1.75 / 6.75))
        self.assertEqual(result.debug.effective_genre_evidence, (("Drama", 1.75),))
        self.assertLess(result.reason_signals[0].strength, 0)

    def test_more_evidence_shrinks_less_and_increases_confidence(self):
        def prediction(count):
            movies = tuple(MovieMetadata(i, str(i), ("Drama",) if i <= count else ("Comedy",))
                           for i in range(1, 2 * count + 1))
            ratings = tuple(UserRating(1, m.movie_id, 5 if m.movie_id <= count else 1)
                            for m in movies)
            profile = build_profile(1, ratings, movies)
            self.assertEqual(profile.baseline, 3)
            self.assertAlmostEqual(profile.preference_for("Drama"), 2 * count / (count + 5))
            return predict_one(profile, CANDIDATE)

        sparse, dense = prediction(1), prediction(10)
        self.assertGreater(dense.predicted_score, sparse.predicted_score)
        self.assertGreater(dense.confidence, sparse.confidence)
        self.assertLess(dense.confidence, 1)

    def test_single_rating_is_neutral_with_low_confidence(self):
        profile = build_profile(1, (UserRating(1, 1, 5),), MOVIES)
        result = predict_one(profile, CANDIDATE)
        self.assertEqual(result.predicted_score, 5)
        self.assertAlmostEqual(result.confidence, 1 / 36)
        self.assertEqual(result.reason_signals, ())

    def test_unknown_and_missing_genres_have_zero_confidence(self):
        profile = build_profile(1, (UserRating(1, 1, 4),), MOVIES)
        for genres in ((), (" ",), ("Unknown",)):
            with self.subTest(genres=genres):
                result = predict_one(profile, MovieMetadata(100, "Candidate", genres))
                self.assertEqual(result.predicted_score, profile.baseline)
                self.assertEqual(result.confidence, 0)
                self.assertEqual(result.reason_signals, ())

    def test_partial_coverage_reduces_confidence_and_adjustment(self):
        ratings = (UserRating(1, 1, 5), UserRating(1, 3, 1))
        profile = build_profile(1, ratings, MOVIES)
        known = predict_one(profile, CANDIDATE)
        partial = predict_one(profile, MovieMetadata(101, "Partial", ("Drama", "Unknown")))
        duplicate = predict_one(profile, MovieMetadata(102, "Duplicate", ("Drama", " DRAMA ")))
        self.assertAlmostEqual(partial.confidence, known.confidence / 2)
        self.assertAlmostEqual(partial.predicted_score - 3, (known.predicted_score - 3) / 2)
        self.assertEqual(duplicate.confidence, known.confidence)
        self.assertEqual(duplicate.predicted_score, known.predicted_score)

    def test_overlapping_genres_do_not_multiply_confidence(self):
        movie = MovieMetadata(1, "Many genres", ("Drama", "Comedy", "Action"))
        profile = build_profile(1, (UserRating(1, 1, 5),), (movie,))
        single = predict_one(profile, CANDIDATE)
        many = predict_one(profile, MovieMetadata(100, "Many", movie.genres))
        self.assertAlmostEqual(single.confidence, many.confidence)

    def test_old_ratings_retain_evidence_and_missing_dates_are_conservative(self):
        profile = build_profile(
            1, (UserRating(1, 1, 5, 0), UserRating(1, 2, 1)), MOVIES,
            ProfileConfig(recency_half_life_days=1, reference_timestamp=10000 * DAY),
        )
        self.assertEqual(profile.effective_evidence_for("Drama"), 1)
        self.assertEqual(profile.evidence_for("Drama"), 2)

    def test_all_missing_dates_have_equal_weights(self):
        profile = build_profile(1, (UserRating(1, 1, 5), UserRating(1, 2, 1)), MOVIES)
        self.assertEqual(profile.effective_evidence_for("Drama"), 2)
        self.assertEqual(profile.preference_for("Drama"), 0)

    def test_recency_can_be_disabled(self):
        profile = build_profile(
            1, (UserRating(1, 1, 5, 0), UserRating(1, 2, 1, 10000 * DAY)), MOVIES,
            ProfileConfig(recency_half_life_days=None),
        )
        self.assertEqual(profile.effective_evidence_for("Drama"), 2)
        self.assertEqual(profile.preference_for("Drama"), 0)

    def test_reference_is_deterministic_and_as_of_boundary_is_enforced(self):
        ratings = (UserRating(1, 1, 5, 0), UserRating(1, 2, 1, 10 * DAY))
        default = build_profile(1, ratings, MOVIES)
        explicit = build_profile(1, ratings, MOVIES, ProfileConfig(reference_timestamp=10 * DAY))
        self.assertEqual(default.genre_preferences, explicit.genre_preferences)
        self.assertEqual(default.metadata.reference_timestamp, 10 * DAY)
        with self.assertRaisesRegex(ValueError, "later than reference"):
            build_profile(1, ratings, MOVIES, ProfileConfig(reference_timestamp=DAY))

    def test_missing_metadata_does_not_fabricate_evidence(self):
        movies = MOVIES + (MovieMetadata(4, "Blank genres", (" ",)),)
        ratings = (UserRating(1, 999, 5), UserRating(1, 4, 1))
        profile = build_profile(1, ratings, movies)
        self.assertEqual(profile.metadata.ratings_without_movie_metadata, 1)
        self.assertEqual(profile.metadata.ratings_without_genres, 1)
        self.assertEqual(profile.metadata.movies_with_genres, 0)
        self.assertEqual(predict_one(profile, CANDIDATE).confidence, 0)

    def test_duplicates_cannot_inflate_evidence(self):
        with self.assertRaisesRegex(ValueError, "Duplicate ratings"):
            build_profile(1, (UserRating(1, 1, 5), UserRating(1, 1, 5)), MOVIES)

    def test_standalone_batch_matches_single_with_generators_and_debug(self):
        profile = build_profile(1, (UserRating(1, 1, 5), UserRating(1, 3, 1)), MOVIES)
        candidates = (CANDIDATE, MovieMetadata(101, "Missing genres"), CANDIDATE)
        config = ScoringConfig(genre_weight=.5)
        batch = predict_batch(profile, iter(candidates), config, include_debug=True)
        self.assertEqual(batch, tuple(predict_one(profile, m, config, True) for m in candidates))
        self.assertTrue(all(isinstance(result, PredictionResult) for result in batch))
        self.assertEqual(predict_batch(profile, iter(())), ())
        self.assertIsInstance(batch[0].to_dict()["confidence"], float)

    def test_model_and_standalone_paths_share_config_and_results(self):
        ratings = (UserRating(1, 1, 5, 0), UserRating(1, 3, 1, DAY))
        config = ProfileConfig(regularization_strength=2, recency_half_life_days=1)
        scoring = ScoringConfig(confidence_prior_count=3)
        model = ContentModel(ratings, MOVIES + (CANDIDATE,), scoring, config)
        profile = build_profile(1, ratings, MOVIES, config)
        expected = predict_one(profile, CANDIDATE, scoring, True)
        self.assertEqual(model.predict_one(1, 100, True), expected)
        self.assertEqual(model.predict((1,), (100,), True), (expected,))
        self.assertIn(expected, model.predict_unseen((1,), include_debug=True))
        self.assertIs(model.build_profile(1), model.build_profile(1))

    def test_disabled_genre_component_has_no_confidence_or_reasons(self):
        profile = build_profile(1, (UserRating(1, 1, 5), UserRating(1, 3, 1)), MOVIES)
        for config in (ScoringConfig(genre_weight=0), ScoringConfig(max_abs_genre_component=0)):
            result = predict_one(profile, CANDIDATE, config)
            self.assertEqual(result.predicted_score, profile.baseline)
            self.assertEqual(result.confidence, 0)
            self.assertEqual(result.reason_signals, ())

    def test_invalid_reliability_settings_are_rejected(self):
        cases = {
            "regularization_strength": (-1, float("nan"), float("inf"), True, "5"),
            "recency_half_life_days": (0, -1, float("nan"), float("inf"), True, "5"),
            "minimum_recency_weight": (0, -1, 1.1, float("nan"), True),
            "reference_timestamp": (-1, 1.5, True),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ProfileConfig(**{name: value})
        for value in (0, -1, float("nan"), float("inf"), True, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ScoringConfig(confidence_prior_count=value)


if __name__ == "__main__":
    unittest.main()
