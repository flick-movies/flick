from src.content.baselines import (
    RatingResidual,
    UserBaseline,
    calculate_user_baseline,
    calculate_user_baselines,
)
from src.content.genres import (
    GenrePreference,
    aggregate_genre_preferences,
    normalized_genre_contributions,
)
from src.content.errors import UnknownMovieError, UnknownUserError
from src.content.model import ContentModel
from src.content.profiles import (
    PROFILE_VERSION,
    ProfileMetadata,
    UserTasteProfile,
    build_profile,
)
from src.content.scoring import (
    GenreComponentResult,
    GenreMatch,
    ScoringConfig,
    genre_component,
    predict_one,
)
from src.content.schemas import (
    MovieMetadata,
    PredictionDebug,
    PredictionResult,
    ReasonSignal,
    UserRating,
)

__all__ = [
    "GenrePreference",
    "GenreComponentResult",
    "GenreMatch",
    "MovieMetadata",
    "PROFILE_VERSION",
    "PredictionDebug",
    "PredictionResult",
    "ProfileMetadata",
    "RatingResidual",
    "ReasonSignal",
    "ScoringConfig",
    "ContentModel",
    "UnknownMovieError",
    "UnknownUserError",
    "UserBaseline",
    "UserRating",
    "UserTasteProfile",
    "aggregate_genre_preferences",
    "build_profile",
    "calculate_user_baseline",
    "calculate_user_baselines",
    "genre_component",
    "normalized_genre_contributions",
    "predict_one",
]
