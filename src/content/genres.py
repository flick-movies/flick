from __future__ import annotations

import math
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
    effective_evidence_count: float | None = None
    regularized_contribution: float | None = None

    @property
    def evidence_weight(self) -> float:
        return (
            float(self.movie_count)
            if self.effective_evidence_count is None
            else self.effective_evidence_count
        )

    @property
    def preference(self) -> float:
        return (
            self.mean_contribution
            if self.regularized_contribution is None
            else self.regularized_contribution
        )


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
    *,
    rating_weights: Mapping[int, float] | None = None,
    regularization_strength: float = 0.0,
) -> tuple[GenrePreference, ...]:
    if not math.isfinite(regularization_strength) or regularization_strength < 0:
        raise ValueError("regularization_strength must be finite and non-negative")
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    evidence: dict[str, float] = {}
    names: dict[str, str] = {}

    for residual in baseline.residuals:
        weight = 1.0 if rating_weights is None else rating_weights[residual.movie_id]
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("rating weights must be finite and in (0, 1]")
        movie = movies_by_id.get(residual.movie_id)
        if movie is None:
            continue

        contributions = normalized_genre_contributions(
            residual.residual,
            movie.genres,
        )

        for genre, contribution in contributions.items():
            key = genre.casefold()
            names[key] = min(names.get(key, genre), genre)
            totals[key] = totals.get(key, 0.0) + weight * contribution
            counts[key] = counts.get(key, 0) + 1
            evidence[key] = evidence.get(key, 0.0) + weight

    return tuple(
        GenrePreference(
            genre=names[genre],
            total_contribution=float(totals[genre]),
            mean_contribution=float(totals[genre] / evidence[genre]),
            movie_count=counts[genre],
            effective_evidence_count=evidence[genre],
            regularized_contribution=totals[genre] / (
                evidence[genre] + regularization_strength
            ),
        )
        for genre in sorted(totals)
    )
