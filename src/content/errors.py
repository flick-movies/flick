class UnknownUserError(LookupError):
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        super().__init__(f"Unknown user ID: {user_id}")


class UnknownMovieError(LookupError):
    def __init__(self, movie_id: int) -> None:
        self.movie_id = movie_id
        super().__init__(f"Unknown movie ID: {movie_id}")
