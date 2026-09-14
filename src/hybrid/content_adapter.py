import pandas as pd

from src.content.model import ContentModel
from src.content.schemas import MovieMetadata, UserRating


def build_content_model_from_frames(
    profile_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> ContentModel:
    ratings_objects = [
        UserRating(
            user_id=int(row.userId),
            movie_id=int(row.movieId),
            rating=float(row.rating),
            timestamp=(
                int(row.timestamp)
                if hasattr(row, "timestamp") and pd.notna(row.timestamp)
                else None
            ),
        )
        for row in profile_ratings.itertuples(index=False)
    ]

    movie_objects = [
        MovieMetadata(
            movie_id=int(row.movieId),
            title=str(row.title),
            genres=tuple(str(row.genres).split("|")),
        )
        for row in movies.itertuples(index=False)
    ]

    return ContentModel(
        ratings=ratings_objects,
        movies=movie_objects,
    )