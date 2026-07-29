from load_data import load_movielens


def explore() -> None:
    ratings, movies = load_movielens("data")

    rated_movie_ids = ratings["movieId"].nunique()
    total_movie_ids = movies["movieId"].nunique()

    ratings_per_movie = ratings.groupby("movieId").size()

    movies_with_one_rating = (ratings_per_movie == 1).sum()
    movies_with_no_ratings = total_movie_ids - rated_movie_ids

    print(f"Movies in movies.csv: {total_movie_ids:,}")
    print(f"Movies with at least one rating: {rated_movie_ids:,}")
    print(f"Movies with no ratings: {movies_with_no_ratings:,}")
    print(f"Movies with exactly one rating: {movies_with_one_rating:,}")

    print(ratings["rating"].value_counts().sort_index())

    print("\nMost-rated movies:")

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            number_of_ratings=("rating", "count"),
        )
        .reset_index()
        .merge(movies, on="movieId")
    )

    print(
        movie_stats
        .sort_values("number_of_ratings", ascending=False)
        [["title", "number_of_ratings", "average_rating"]]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    explore()