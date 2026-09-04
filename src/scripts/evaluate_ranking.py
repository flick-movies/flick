from src.evaluation.ranking import (
    evaluate_pairwise_accuracy,
)

from src.evaluation.reports import (
    print_confidence_intervals,
    print_correlations,
    print_effect_size_counts,
    print_extreme_users,
    print_history_analysis,
    print_history_win_rates,
    print_improvement_distribution,
    print_overall_results,
    print_pair_count_analysis,
    print_sanity_checks,
    print_user_win_analysis,
    print_weighted_results,
)

from src.load_data import load_movielens


def main() -> None:
    ratings, movies = load_movielens("data")

    print("Dataset")
    print("=" * 72)

    print(f"Ratings:      {len(ratings):,}")
    print(
        f"Users:        "
        f"{ratings['userId'].nunique():,}"
    )
    print(
        f"Rated movies: "
        f"{ratings['movieId'].nunique():,}"
    )
    print(f"Movie rows:   {len(movies):,}")

    print()

    result = evaluate_pairwise_accuracy(
        ratings=ratings,
        movies=movies,
    )

    print_overall_results(result)

    user_results = result.user_results

    print_weighted_results(user_results)
    print_history_analysis(user_results)
    print_user_win_analysis(user_results)
    print_history_win_rates(user_results)
    print_pair_count_analysis(user_results)
    print_improvement_distribution(user_results)
    print_effect_size_counts(user_results)
    print_confidence_intervals(user_results)
    print_correlations(user_results)
    print_extreme_users(user_results)
    print_sanity_checks(result)


if __name__ == "__main__":
    main()