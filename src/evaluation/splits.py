import pandas as pd

from src.hybrid.ml_reranker import chronological_split


def build_user_evaluation_split(
    user_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile, train, test = chronological_split(user_ratings)
    return profile, train, test