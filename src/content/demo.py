import csv
from pathlib import Path

from src.content.baselines import calculate_user_baseline
from src.content.genres import aggregate_genre_preferences
from src.data_processing.movielens import (
    movie_metadata_from_records,
    user_ratings_from_records,
)


def _read_records(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(csv.DictReader(source))


def main(user_id: int = 1, data_directory: str | Path = "data") -> None:
    directory = Path(data_directory)
    rating_rows = _read_records(directory / "ratings.csv")
    movie_rows = _read_records(directory / "movies.csv")
    user_rows = tuple(
        row for row in rating_rows if int(row["userId"]) == user_id
    )
    ratings = user_ratings_from_records(user_rows)
    movies = movie_metadata_from_records(movie_rows)
    movies_by_id = {movie.movie_id: movie for movie in movies}
    baseline = calculate_user_baseline(ratings)
    preferences = aggregate_genre_preferences(baseline, movies_by_id)

    print(f"User {user_id}")
    print(f"Ratings: {baseline.rating_count}")
    print(f"Mean rating: {baseline.mean_rating:.3f}")
    print()
    print(f"{'Genre':<20} {'Movies':>6} {'Total':>10} {'Mean':>10}")

    for preference in sorted(
        preferences,
        key=lambda item: item.total_contribution,
        reverse=True,
    ):
        print(
            f"{preference.genre:<20} "
            f"{preference.movie_count:>6} "
            f"{preference.total_contribution:>+10.3f} "
            f"{preference.mean_contribution:>+10.3f}"
        )


if __name__ == "__main__":
    main()
