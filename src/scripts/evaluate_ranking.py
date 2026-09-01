from __future__ import annotations

import numpy as np

from src.evaluation.ranking import (
    PairwiseEvaluation,
    UserPairwiseResult,
    evaluate_pairwise_accuracy,
)
from src.load_data import load_movielens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(values))


def weighted_accuracy(
    user_results: list[UserPairwiseResult],
    attribute: str,
) -> float:
    """
    Compute pair-weighted accuracy.

    The main benchmark macro-averages users, meaning every user has equal
    weight regardless of how many test pairs they contribute.

    This metric instead weights users by pairs_evaluated.
    """
    total_pairs = sum(
        user_result.pairs_evaluated
        for user_result in user_results
    )

    if total_pairs == 0:
        return float("nan")

    weighted_correct = sum(
        getattr(user_result, attribute)
        * user_result.pairs_evaluated
        for user_result in user_results
    )

    return weighted_correct / total_pairs


def bootstrap_mean_difference(
    differences: list[float],
    iterations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Estimate a 95% bootstrap confidence interval for a mean difference.

    Resampling happens at the user level because the benchmark itself
    macro-averages users.
    """
    if not differences:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    differences_array = np.asarray(differences, dtype=float)

    bootstrap_means = np.empty(iterations)

    for i in range(iterations):
        sample = rng.choice(
            differences_array,
            size=len(differences_array),
            replace=True,
        )
        bootstrap_means[i] = np.mean(sample)

    lower = float(np.percentile(bootstrap_means, 2.5))
    upper = float(np.percentile(bootstrap_means, 97.5))

    return lower, upper


def print_comparison_counts(
    user_results: list[UserPairwiseResult],
    first_attribute: str,
    second_attribute: str,
    first_name: str,
    second_name: str,
) -> None:
    better = 0
    worse = 0
    ties = 0

    for user_result in user_results:
        first_score = getattr(user_result, first_attribute)
        second_score = getattr(user_result, second_attribute)

        if first_score > second_score:
            better += 1
        elif first_score < second_score:
            worse += 1
        else:
            ties += 1

    total = len(user_results)

    print(f"{first_name} better: {better:>3} ({better / total:.1%})")
    print(f"{second_name} better: {worse:>3} ({worse / total:.1%})")
    print(f"Tied:           {ties:>3} ({ties / total:.1%})")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def print_overall_results(result: PairwiseEvaluation) -> None:
    print("Pairwise Ranking Evaluation")
    print("=" * 72)

    print(f"Users evaluated: {result.users_evaluated}")
    print(f"Test pairs:      {result.pairs_evaluated:,}")

    print()
    print("Macro Accuracy")
    print("-" * 72)
    print(f"Baseline:   {result.baseline_accuracy:.3%}")
    print(f"Heuristic:  {result.heuristic_accuracy:.3%}")
    print(f"ML:         {result.ml_accuracy:.3%}")

    print()
    print("ML improvement:")
    print(
        f"  vs baseline:  "
        f"{result.ml_accuracy - result.baseline_accuracy:+.3%}"
    )
    print(
        f"  vs heuristic: "
        f"{result.ml_accuracy - result.heuristic_accuracy:+.3%}"
    )


# ---------------------------------------------------------------------------
# Macro vs pair-weighted accuracy
# ---------------------------------------------------------------------------

def print_weighted_results(
    user_results: list[UserPairwiseResult],
) -> None:
    baseline = weighted_accuracy(
        user_results,
        "baseline_accuracy",
    )
    heuristic = weighted_accuracy(
        user_results,
        "heuristic_accuracy",
    )
    ml = weighted_accuracy(
        user_results,
        "ml_accuracy",
    )

    print()
    print("Pair-Weighted Accuracy")
    print("=" * 72)
    print(
        "Unlike the main benchmark, users with more test pairs "
        "receive more weight."
    )
    print()

    print(f"Baseline:   {baseline:.3%}")
    print(f"Heuristic:  {heuristic:.3%}")
    print(f"ML:         {ml:.3%}")

    print()
    print(f"ML vs baseline:  {ml - baseline:+.3%}")
    print(f"ML vs heuristic: {ml - heuristic:+.3%}")


# ---------------------------------------------------------------------------
# User history analysis
# ---------------------------------------------------------------------------

def print_history_analysis(
    user_results: list[UserPairwiseResult],
) -> None:
    buckets = [
        (10, 29),
        (30, 49),
        (50, 99),
        (100, 199),
        (200, float("inf")),
    ]

    print()
    print("Accuracy by User History")
    print("=" * 72)

    print(
        f"{'Ratings':>10} | "
        f"{'Users':>5} | "
        f"{'Baseline':>9} | "
        f"{'Heuristic':>9} | "
        f"{'ML':>9} | "
        f"{'ML-Base':>9}"
    )
    print("-" * 72)

    for lower, upper in buckets:
        bucket_results = [
            user_result
            for user_result in user_results
            if lower <= user_result.rating_count <= upper
        ]

        if not bucket_results:
            continue

        baseline = mean([
            user_result.baseline_accuracy
            for user_result in bucket_results
        ])

        heuristic = mean([
            user_result.heuristic_accuracy
            for user_result in bucket_results
        ])

        ml = mean([
            user_result.ml_accuracy
            for user_result in bucket_results
        ])

        label = (
            f"{lower}+"
            if upper == float("inf")
            else f"{lower}-{int(upper)}"
        )

        print(
            f"{label:>10} | "
            f"{len(bucket_results):>5} | "
            f"{baseline:>8.3%} | "
            f"{heuristic:>8.3%} | "
            f"{ml:>8.3%} | "
            f"{ml - baseline:>+8.3%}"
        )


# ---------------------------------------------------------------------------
# Win / loss analysis
# ---------------------------------------------------------------------------

def print_user_win_analysis(
    user_results: list[UserPairwiseResult],
) -> None:
    print()
    print("Per-User Win Rates")
    print("=" * 72)

    print("ML vs Baseline")
    print("-" * 30)
    print_comparison_counts(
        user_results,
        first_attribute="ml_accuracy",
        second_attribute="baseline_accuracy",
        first_name="ML",
        second_name="Baseline",
    )

    print()
    print("ML vs Heuristic")
    print("-" * 30)
    print_comparison_counts(
        user_results,
        first_attribute="ml_accuracy",
        second_attribute="heuristic_accuracy",
        first_name="ML",
        second_name="Heuristic",
    )


# ---------------------------------------------------------------------------
# Win rate by history size
# ---------------------------------------------------------------------------

def print_history_win_rates(
    user_results: list[UserPairwiseResult],
) -> None:
    buckets = [
        (10, 29),
        (30, 49),
        (50, 99),
        (100, 199),
        (200, float("inf")),
    ]

    print()
    print("ML vs Baseline Win Rate by User History")
    print("=" * 72)

    print(
        f"{'Ratings':>10} | "
        f"{'Users':>5} | "
        f"{'ML Wins':>8} | "
        f"{'Base Wins':>9} | "
        f"{'Ties':>5} | "
        f"{'ML Win %':>8}"
    )
    print("-" * 72)

    for lower, upper in buckets:
        bucket_results = [
            user_result
            for user_result in user_results
            if lower <= user_result.rating_count <= upper
        ]

        if not bucket_results:
            continue

        ml_wins = sum(
            user_result.ml_accuracy > user_result.baseline_accuracy
            for user_result in bucket_results
        )

        baseline_wins = sum(
            user_result.ml_accuracy < user_result.baseline_accuracy
            for user_result in bucket_results
        )

        ties = len(bucket_results) - ml_wins - baseline_wins

        label = (
            f"{lower}+"
            if upper == float("inf")
            else f"{lower}-{int(upper)}"
        )

        print(
            f"{label:>10} | "
            f"{len(bucket_results):>5} | "
            f"{ml_wins:>8} | "
            f"{baseline_wins:>9} | "
            f"{ties:>5} | "
            f"{ml_wins / len(bucket_results):>7.1%}"
        )


# ---------------------------------------------------------------------------
# Pair-count reliability analysis
# ---------------------------------------------------------------------------

def print_pair_count_analysis(
    user_results: list[UserPairwiseResult],
) -> None:
    """
    Accuracy for a user with only a handful of test pairs can be noisy.

    This shows whether conclusions change when users contribute more
    evaluated comparisons.
    """
    buckets = [
        (1, 24),
        (25, 99),
        (100, 499),
        (500, 1999),
        (2000, float("inf")),
    ]

    print()
    print("Accuracy by Number of Evaluated Pairs")
    print("=" * 72)

    print(
        f"{'Pairs':>10} | "
        f"{'Users':>5} | "
        f"{'Baseline':>9} | "
        f"{'Heuristic':>9} | "
        f"{'ML':>9} | "
        f"{'ML-Base':>9}"
    )
    print("-" * 72)

    for lower, upper in buckets:
        bucket_results = [
            user_result
            for user_result in user_results
            if lower <= user_result.pairs_evaluated <= upper
        ]

        if not bucket_results:
            continue

        baseline = mean([
            user_result.baseline_accuracy
            for user_result in bucket_results
        ])

        heuristic = mean([
            user_result.heuristic_accuracy
            for user_result in bucket_results
        ])

        ml = mean([
            user_result.ml_accuracy
            for user_result in bucket_results
        ])

        label = (
            f"{lower}+"
            if upper == float("inf")
            else f"{lower}-{int(upper)}"
        )

        print(
            f"{label:>10} | "
            f"{len(bucket_results):>5} | "
            f"{baseline:>8.3%} | "
            f"{heuristic:>8.3%} | "
            f"{ml:>8.3%} | "
            f"{ml - baseline:>+8.3%}"
        )


# ---------------------------------------------------------------------------
# Improvement distribution
# ---------------------------------------------------------------------------

def print_improvement_distribution(
    user_results: list[UserPairwiseResult],
) -> None:
    baseline_differences = [
        user_result.ml_accuracy - user_result.baseline_accuracy
        for user_result in user_results
    ]

    heuristic_differences = [
        user_result.ml_accuracy - user_result.heuristic_accuracy
        for user_result in user_results
    ]

    print()
    print("Distribution of ML Improvement")
    print("=" * 72)

    print("ML minus Baseline")
    print("-" * 30)
    print(f"Mean:   {mean(baseline_differences):+.3%}")
    print(f"Median: {median(baseline_differences):+.3%}")

    for percentile in [10, 25, 50, 75, 90]:
        value = float(
            np.percentile(
                baseline_differences,
                percentile,
            )
        )

        print(
            f"P{percentile:<2}:    "
            f"{value:+.3%}"
        )

    print()
    print("ML minus Heuristic")
    print("-" * 30)
    print(f"Mean:   {mean(heuristic_differences):+.3%}")
    print(f"Median: {median(heuristic_differences):+.3%}")

    for percentile in [10, 25, 50, 75, 90]:
        value = float(
            np.percentile(
                heuristic_differences,
                percentile,
            )
        )

        print(
            f"P{percentile:<2}:    "
            f"{value:+.3%}"
        )


# ---------------------------------------------------------------------------
# Meaningful wins / losses
# ---------------------------------------------------------------------------

def print_effect_size_counts(
    user_results: list[UserPairwiseResult],
) -> None:
    differences = [
        user_result.ml_accuracy - user_result.baseline_accuracy
        for user_result in user_results
    ]

    print()
    print("Magnitude of ML vs Baseline Changes")
    print("=" * 72)

    thresholds = [
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    for threshold in thresholds:
        improved = sum(
            difference >= threshold
            for difference in differences
        )

        worsened = sum(
            difference <= -threshold
            for difference in differences
        )

        print(
            f"At least {threshold:.0%} better: "
            f"{improved:>3} users "
            f"({improved / len(differences):.1%})"
        )

        print(
            f"At least {threshold:.0%} worse:  "
            f"{worsened:>3} users "
            f"({worsened / len(differences):.1%})"
        )

        print()


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def print_confidence_intervals(
    user_results: list[UserPairwiseResult],
) -> None:
    baseline_differences = [
        user_result.ml_accuracy - user_result.baseline_accuracy
        for user_result in user_results
    ]

    heuristic_differences = [
        user_result.ml_accuracy - user_result.heuristic_accuracy
        for user_result in user_results
    ]

    baseline_lower, baseline_upper = bootstrap_mean_difference(
        baseline_differences
    )

    heuristic_lower, heuristic_upper = bootstrap_mean_difference(
        heuristic_differences
    )

    print()
    print("User-Level Bootstrap 95% Confidence Intervals")
    print("=" * 72)

    print(
        "ML vs baseline mean difference: "
        f"{mean(baseline_differences):+.3%}"
    )

    print(
        "95% CI: "
        f"[{baseline_lower:+.3%}, {baseline_upper:+.3%}]"
    )

    print()

    print(
        "ML vs heuristic mean difference: "
        f"{mean(heuristic_differences):+.3%}"
    )

    print(
        "95% CI: "
        f"[{heuristic_lower:+.3%}, {heuristic_upper:+.3%}]"
    )


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def print_correlations(
    user_results: list[UserPairwiseResult],
) -> None:
    """
    Correlation does not establish causation.

    These numbers are diagnostics for whether users with more history or
    more test evidence tend to receive larger ML improvements.
    """
    rating_counts = np.asarray([
        user_result.rating_count
        for user_result in user_results
    ])

    pair_counts = np.asarray([
        user_result.pairs_evaluated
        for user_result in user_results
    ])

    improvements = np.asarray([
        user_result.ml_accuracy - user_result.baseline_accuracy
        for user_result in user_results
    ])

    rating_correlation = float(
        np.corrcoef(
            rating_counts,
            improvements,
        )[0, 1]
    )

    pair_correlation = float(
        np.corrcoef(
            pair_counts,
            improvements,
        )[0, 1]
    )

    print()
    print("Correlation Diagnostics")
    print("=" * 72)

    print(
        "Rating count vs ML improvement: "
        f"{rating_correlation:+.3f}"
    )

    print(
        "Evaluated pairs vs ML improvement: "
        f"{pair_correlation:+.3f}"
    )

    print()
    print(
        "These are diagnostic correlations only; "
        "they do not imply causation."
    )


# ---------------------------------------------------------------------------
# Best / worst users
# ---------------------------------------------------------------------------

def print_extreme_users(
    user_results: list[UserPairwiseResult],
    limit: int = 10,
) -> None:
    sorted_results = sorted(
        user_results,
        key=lambda user_result:
            user_result.ml_accuracy
            - user_result.baseline_accuracy,
        reverse=True,
    )

    print()
    print("Largest ML Improvements over Baseline")
    print("=" * 72)

    print(
        f"{'User':>6} | "
        f"{'Ratings':>7} | "
        f"{'Pairs':>7} | "
        f"{'Baseline':>9} | "
        f"{'ML':>9} | "
        f"{'Delta':>9}"
    )
    print("-" * 72)

    for user_result in sorted_results[:limit]:
        difference = (
            user_result.ml_accuracy
            - user_result.baseline_accuracy
        )

        print(
            f"{user_result.user_id:>6} | "
            f"{user_result.rating_count:>7} | "
            f"{user_result.pairs_evaluated:>7} | "
            f"{user_result.baseline_accuracy:>8.3%} | "
            f"{user_result.ml_accuracy:>8.3%} | "
            f"{difference:>+8.3%}"
        )

    print()
    print("Largest ML Regressions vs Baseline")
    print("=" * 72)

    print(
        f"{'User':>6} | "
        f"{'Ratings':>7} | "
        f"{'Pairs':>7} | "
        f"{'Baseline':>9} | "
        f"{'ML':>9} | "
        f"{'Delta':>9}"
    )
    print("-" * 72)

    for user_result in sorted_results[-limit:][::-1]:
        difference = (
            user_result.ml_accuracy
            - user_result.baseline_accuracy
        )

        print(
            f"{user_result.user_id:>6} | "
            f"{user_result.rating_count:>7} | "
            f"{user_result.pairs_evaluated:>7} | "
            f"{user_result.baseline_accuracy:>8.3%} | "
            f"{user_result.ml_accuracy:>8.3%} | "
            f"{difference:>+8.3%}"
        )


# ---------------------------------------------------------------------------
# Data sanity checks
# ---------------------------------------------------------------------------

def print_sanity_checks(
    result: PairwiseEvaluation,
) -> None:
    user_results = result.user_results

    print()
    print("Evaluation Sanity Checks")
    print("=" * 72)

    checks = {
        "User result count matches users evaluated":
            len(user_results) == result.users_evaluated,

        "Per-user pair counts sum to total":
            sum(
                user_result.pairs_evaluated
                for user_result in user_results
            ) == result.pairs_evaluated,

        "Every user has at least one evaluated pair":
            all(
                user_result.pairs_evaluated > 0
                for user_result in user_results
            ),

        "Every user has at least 10 ratings":
            all(
                user_result.rating_count >= 10
                for user_result in user_results
            ),

        "Baseline accuracies within [0, 1]":
            all(
                0.0 <= user_result.baseline_accuracy <= 1.0
                for user_result in user_results
            ),

        "Heuristic accuracies within [0, 1]":
            all(
                0.0 <= user_result.heuristic_accuracy <= 1.0
                for user_result in user_results
            ),

        "ML accuracies within [0, 1]":
            all(
                0.0 <= user_result.ml_accuracy <= 1.0
                for user_result in user_results
            ),
        "Matrix factorization accuracies within [0, 1]":
            all(
                0.0 <= user_result.matrix_factorization_accuracy <= 1.0
                for user_result in user_results
            ),
    }

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ratings, movies = load_movielens("data")

    print("Dataset")
    print("=" * 72)
    print(f"Ratings:      {len(ratings):,}")
    print(f"Users:        {ratings['userId'].nunique():,}")
    print(f"Rated movies: {ratings['movieId'].nunique():,}")
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

    print(f"Baseline:   {result.baseline_accuracy:.3%}")
    print(f"Heuristic:  {result.heuristic_accuracy:.3%}")
    print(f"ML:         {result.ml_accuracy:.3%}")
    print(
        f"Matrix MF:  "
        f"{result.matrix_factorization_accuracy:.3%}"
    )


if __name__ == "__main__":
    main()