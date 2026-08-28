from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.content.baselines import calculate_user_baseline
from src.content.errors import UnknownUserError
from src.content.genres import GenrePreference, aggregate_genre_preferences
from src.content.schemas import MovieMetadata, UserRating


PROFILE_VERSION = "genre-v1"


@dataclass(frozen=True)
class ProfileMetadata:
    ratings_used: int
    movies_with_metadata: int
    movies_with_genres: int
    ratings_without_movie_metadata: int
    ratings_without_genres: int


@dataclass(frozen=True)
class UserTasteProfile:
    user_id: int
    baseline: float
    rating_count: int
    genre_preferences: tuple[GenrePreference, ...]
    profile_version: str
    metadata: ProfileMetadata

    def preference_for(self, genre: str) -> float:
        key = genre.strip().casefold()
        for preference in self.genre_preferences:
            if preference.genre.casefold() == key:
                return preference.mean_contribution
        return 0.0

    def evidence_for(self, genre: str) -> int:
        key = genre.strip().casefold()
        for preference in self.genre_preferences:
            if preference.genre.casefold() == key:
                return preference.movie_count
        return 0


def _index_movies(
    movies: Mapping[int, MovieMetadata] | Iterable[MovieMetadata],
) -> dict[int, MovieMetadata]:
    if isinstance(movies, Mapping):
        return dict(movies)

    indexed: dict[int, MovieMetadata] = {}
    for movie in movies:
        if movie.movie_id in indexed:
            raise ValueError(f"Duplicate movie ID: {movie.movie_id}")
        indexed[movie.movie_id] = movie
    return indexed


def build_profile(
    user_id: int,
    ratings: Iterable[UserRating],
    movies: Mapping[int, MovieMetadata] | Iterable[MovieMetadata],
) -> UserTasteProfile:
    user_ratings = tuple(
        rating for rating in ratings if rating.user_id == user_id
    )
    if not user_ratings:
        raise UnknownUserError(user_id)

    movies_by_id = _index_movies(movies)
    baseline = calculate_user_baseline(user_ratings)
    preferences = aggregate_genre_preferences(baseline, movies_by_id)
    movies_with_metadata = sum(
        rating.movie_id in movies_by_id for rating in user_ratings
    )
    movies_with_genres = sum(
        bool(movies_by_id[rating.movie_id].genres)
        for rating in user_ratings
        if rating.movie_id in movies_by_id
    )

    metadata = ProfileMetadata(
        ratings_used=len(user_ratings),
        movies_with_metadata=movies_with_metadata,
        movies_with_genres=movies_with_genres,
        ratings_without_movie_metadata=len(user_ratings) - movies_with_metadata,
        ratings_without_genres=movies_with_metadata - movies_with_genres,
    )

    return UserTasteProfile(
        user_id=user_id,
        baseline=baseline.mean_rating,
        rating_count=baseline.rating_count,
        genre_preferences=preferences,
        profile_version=PROFILE_VERSION,
        metadata=metadata,
    )
