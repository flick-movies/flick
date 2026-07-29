from src.genre_recommender import recommend_by_genre
from src.load_data import load_movielens


def main() -> None:
    ratings, movies = load_movielens("data")

    user_id = 106

    recommendations = recommend_by_genre(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        limit=5,
    )

    print(f"Top recommendations for User {user_id}:\n")

    for position, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        print(
            f"{position}. {recommendation.title}\n"
            f"   Genres: {recommendation.genres}\n"
            f"   Predicted rating: "
            f"{recommendation.predicted_rating:.2f}/5\n"
            f"   Personal genre score: "
            f"{recommendation.personal_score:+.3f}\n"
            f"   {recommendation.explanation}\n"
        )


if __name__ == "__main__":
    main()