from src.content.schemas import MovieMetadata, UserRating


TOY_MOVIES = (
    MovieMetadata(1, "Space Frontier", ("Sci-Fi", "Adventure"), release_year=2010),
    MovieMetadata(2, "Galaxy War", ("Sci-Fi", "Action"), release_year=2012),
    MovieMetadata(3, "Quiet Letters", ("Romance", "Drama"), release_year=2008),
    MovieMetadata(4, "Wedding Weekend", ("Romance", "Comedy"), release_year=2015),
    MovieMetadata(5, "Haunted Manor", ("Horror", "Thriller"), release_year=2005),
    MovieMetadata(6, "Laugh Track", ("Comedy",), release_year=2018),
    MovieMetadata(7, "Courtroom Truth", ("Drama",), release_year=2001),
    MovieMetadata(8, "Deep Orbit", ("Sci-Fi",), release_year=2020),
    MovieMetadata(9, "Untitled Archive", (), release_year=None),
    MovieMetadata(
        10,
        "Five Worlds",
        ("Action", "Adventure", "Comedy", "Drama", "Sci-Fi"),
        release_year=2022,
    ),
)

TOY_RATINGS = (
    UserRating(1, 1, 5.0, 100),
    UserRating(1, 2, 4.5, 200),
    UserRating(1, 3, 1.5, 300),
    UserRating(1, 4, 2.0, 400),
    UserRating(2, 1, 1.5, 100),
    UserRating(2, 3, 4.5, 200),
    UserRating(2, 4, 5.0, 300),
    UserRating(2, 6, 4.0, 400),
    UserRating(3, 5, 4.0, 100),
    UserRating(3, 6, 5.0, 200),
    UserRating(3, 7, 4.5, 300),
    UserRating(3, 8, 5.0, 400),
)

TOY_MOVIES_BY_ID = {movie.movie_id: movie for movie in TOY_MOVIES}
