from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.content.schemas import UserRating


@dataclass(frozen=True)
class RatingResidual:
    user_id: int
    movie_id: int
    rating: float
    user_mean: float
    residual: float


@dataclass(frozen=True)
class UserBaseline:
    user_id: int
    rating_count: int
    mean_rating: float
    residuals: tuple[RatingResidual, ...]


def calculate_user_baseline(ratings: Iterable[UserRating]) -> UserBaseline:
    user_ratings = tuple(ratings)
    if not user_ratings:
        raise ValueError("at least one rating is required")

    user_ids = {rating.user_id for rating in user_ratings}
    if len(user_ids) != 1:
        raise ValueError("all ratings must belong to the same user")

    user_id = next(iter(user_ids))
    mean_rating = sum(rating.rating for rating in user_ratings) / len(user_ratings)
    residuals = tuple(
        RatingResidual(
            user_id=rating.user_id,
            movie_id=rating.movie_id,
            rating=float(rating.rating),
            user_mean=float(mean_rating),
            residual=float(rating.rating - mean_rating),
        )
        for rating in user_ratings
    )

    return UserBaseline(
        user_id=user_id,
        rating_count=len(user_ratings),
        mean_rating=float(mean_rating),
        residuals=residuals,
    )


def calculate_user_baselines(
    ratings: Iterable[UserRating],
) -> dict[int, UserBaseline]:
    grouped: dict[int, list[UserRating]] = {}
    for rating in ratings:
        grouped.setdefault(rating.user_id, []).append(rating)

    return {
        user_id: calculate_user_baseline(user_ratings)
        for user_id, user_ratings in sorted(grouped.items())
    }
