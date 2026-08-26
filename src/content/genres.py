from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from src.content.baselines import UserBaseline
from src.content.schemas import MovieMetadata


@dataclass(frozen=True)
class GenrePreference:
    genre: str
    total_contribution: float
    mean_contribution: float
    movie_count: int


def _unique_genres(genres: Iterable[str]) -> tuple[str, ...]:
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


def normalized_genre_contributions(
    residual: float,
    genres: Iterable[str],
) -> dict[str, float]:
    unique_genres = _unique_genres(genres)
    if not unique_genres:
        return {}

    contribution = float(residual) / len(unique_genres)
    return {genre: contribution for genre in unique_genres}


def aggregate_genre_preferences(
    baseline: UserBaseline,
    movies_by_id: Mapping[int, MovieMetadata],
) -> tuple[GenrePreference, ...]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for residual in baseline.residuals:
        movie = movies_by_id.get(residual.movie_id)
        if movie is None:
            continue

        contributions = normalized_genre_contributions(
            residual.residual,
            movie.genres,
        )

        for genre, contribution in contributions.items():
            totals[genre] = totals.get(genre, 0.0) + contribution
            counts[genre] = counts.get(genre, 0) + 1

    return tuple(
        GenrePreference(
            genre=genre,
            total_contribution=float(totals[genre]),
            mean_contribution=float(totals[genre] / counts[genre]),
            movie_count=counts[genre],
        )
        for genre in sorted(totals)
    )
