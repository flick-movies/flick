from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from src.content.schemas import MovieMetadata, UserRating


YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")
NO_GENRES = "(no genres listed)"


def parse_genres(value: object) -> tuple[str, ...]:
    if value is None or not isinstance(value, str):
        return ()

    unique: list[str] = []
    seen: set[str] = set()

    for raw_genre in value.split("|"):
        genre = raw_genre.strip()
        key = genre.casefold()
        if not genre or key == NO_GENRES.casefold() or key in seen:
            continue
        seen.add(key)
        unique.append(genre)

    return tuple(unique)


def _optional_timestamp(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def movie_metadata_from_record(record: Mapping[str, Any]) -> MovieMetadata:
    title = str(record["title"])
    match = YEAR_PATTERN.search(title)
    release_year = int(match.group(1)) if match else None

    return MovieMetadata(
        movie_id=int(record["movieId"]),
        title=title,
        genres=parse_genres(record.get("genres")),
        release_year=release_year,
    )


def movie_metadata_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[MovieMetadata, ...]:
    return tuple(movie_metadata_from_record(record) for record in records)


def user_rating_from_record(record: Mapping[str, Any]) -> UserRating:
    return UserRating(
        user_id=int(record["userId"]),
        movie_id=int(record["movieId"]),
        rating=float(record["rating"]),
        timestamp=_optional_timestamp(record.get("timestamp")),
    )


def user_ratings_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[UserRating, ...]:
    return tuple(user_rating_from_record(record) for record in records)
