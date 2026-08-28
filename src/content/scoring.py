from __future__ import annotations

import math
from dataclasses import dataclass

from src.content.profiles import UserTasteProfile
from src.content.schemas import MovieMetadata, PredictionDebug, PredictionResult


@dataclass(frozen=True)
class ScoringConfig:
    genre_weight: float = 1.0
    max_abs_genre_component: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.genre_weight) or self.genre_weight < 0:
            raise ValueError("genre_weight must be finite and non-negative")
        if (
            not math.isfinite(self.max_abs_genre_component)
            or self.max_abs_genre_component < 0
        ):
            raise ValueError(
                "max_abs_genre_component must be finite and non-negative"
            )


@dataclass(frozen=True)
class GenreMatch:
    genre: str
    preference: float
    evidence_count: int


@dataclass(frozen=True)
class GenreComponentResult:
    raw_contribution: float
    bounded_contribution: float
    matches: tuple[GenreMatch, ...]
    unknown_genres: tuple[str, ...]


def _unique_genres(genres: tuple[str, ...]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for genre in genres:
        cleaned = genre.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return tuple(unique)


def genre_component(
    movie: MovieMetadata,
    profile: UserTasteProfile,
    max_abs_contribution: float = 1.0,
) -> GenreComponentResult:
    if not math.isfinite(max_abs_contribution) or max_abs_contribution < 0:
        raise ValueError("max_abs_contribution must be finite and non-negative")

    movie_genres = _unique_genres(movie.genres)
    preference_map = {
        preference.genre.casefold(): preference
        for preference in profile.genre_preferences
    }
    matches: list[GenreMatch] = []
    unknown_genres: list[str] = []
    contributions: list[float] = []

    for genre in movie_genres:
        preference = preference_map.get(genre.casefold())
        if preference is None:
            unknown_genres.append(genre)
            contributions.append(0.0)
            continue

        matches.append(
            GenreMatch(
                genre=genre,
                preference=preference.mean_contribution,
                evidence_count=preference.movie_count,
            )
        )
        contributions.append(preference.mean_contribution)

    raw_contribution = (
        float(sum(contributions) / len(contributions))
        if contributions
        else 0.0
    )
    bounded_contribution = max(
        -max_abs_contribution,
        min(max_abs_contribution, raw_contribution),
    )

    return GenreComponentResult(
        raw_contribution=raw_contribution,
        bounded_contribution=float(bounded_contribution),
        matches=tuple(matches),
        unknown_genres=tuple(unknown_genres),
    )


def predict_one(
    profile: UserTasteProfile,
    movie: MovieMetadata,
    config: ScoringConfig | None = None,
    include_debug: bool = False,
) -> PredictionResult:
    active_config = config or ScoringConfig()
    component = genre_component(
        movie,
        profile,
        active_config.max_abs_genre_component,
    )
    weighted_adjustment = (
        active_config.genre_weight * component.bounded_contribution
    )
    unclamped_score = profile.baseline + weighted_adjustment
    predicted_score = max(0.0, min(5.0, unclamped_score))
    debug = None

    if include_debug:
        debug = PredictionDebug(
            baseline=profile.baseline,
            movie_genres=_unique_genres(movie.genres),
            matched_genres=tuple(match.genre for match in component.matches),
            unknown_genres=component.unknown_genres,
            raw_genre_component=component.raw_contribution,
            bounded_genre_component=component.bounded_contribution,
            genre_weight=active_config.genre_weight,
            weighted_genre_adjustment=weighted_adjustment,
            unclamped_score=unclamped_score,
            was_clamped=predicted_score != unclamped_score,
        )

    return PredictionResult(
        user_id=profile.user_id,
        movie_id=movie.movie_id,
        predicted_score=float(predicted_score),
        confidence=0.0,
        debug=debug,
    )
