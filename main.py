from src.load_data import load_movielens
from src.hybrid.ml_reranker import recommend_with_ml

def main() -> None:
    ratings, movies = load_movielens("data")

    user_id = 23

    recommendations = recommend_with_ml(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        limit=5,
    )

    print(f"ML recommendations for User {user_id}:\n")

    for position, movie in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"{position}. {movie.title}\n"
            f"   Genres: {movie.genres}\n"
            f"   ML score: {movie.ml_score:.3f}\n"
            f"   Personal score: {movie.personal_score:+.3f}\n"
            f"   Quality score: {movie.quality_score:+.3f}\n"
            f"   Popularity: {movie.popularity:.3f}\n"
        )


if __name__ == "__main__":
    main()