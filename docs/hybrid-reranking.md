# Hybrid and Reranking

## Purpose

The hybrid recommendation system combines personalized user-taste signals with global movie-quality and popularity signals to rank movies a user is likely to prefer.

The current system has two ranking stages:

1. A heuristic genre-based recommender that produces personalized movie features.
2. A learned pairwise logistic-regression reranker that learns how strongly each feature should affect ranking.

The final ML recommender ranks unseen movies using the learned model.

Matrix factorization is currently evaluated as a separate collaborative benchmark. It is not yet used directly as an input feature to the ML reranker.

## Inputs

The system currently uses the MovieLens latest-small dataset.

Required inputs include:

* User ID
* User ratings
* Rating timestamps
* Movie IDs
* Movie titles
* Movie genres
* Ratings from the overall user population

The ML reranker also loads the trained model stored at:

`src/models/ml_reranker.joblib`

## Outputs

The ML recommender returns ranked movies containing:

* `movie_id`
* `title`
* `genres`
* `ml_score`
* `personal_score`
* `quality_score`
* `popularity`

A higher `ml_score` means the trained reranker prefers that movie more strongly.

The ML score is a ranking score and should not be interpreted directly as a predicted 1–5 star rating.

## Core Idea

The system summarizes several useful signals about each movie for a particular user.

The current ML model uses three features:

1. Personal genre score
2. Movie quality score
3. Movie popularity

Instead of manually deciding how much each feature should matter, the system learns their relative importance from users' historical rating preferences.

The model is trained using pairwise comparisons.

For two movies A and B:

* If the user rated A higher than B, the feature difference `A - B` receives label `1`.
* The reverse feature difference `B - A` receives label `0`.

A logistic regression model learns which feature differences tend to correspond to the movie a user preferred.

At recommendation time, every unseen candidate receives the same features and the trained model produces a ranking score.

## Algorithm Flow

### 1. Build the user's historical profile

The user's ratings are collected and ordered using timestamps.

More recent ratings receive greater weight through exponential decay.

The current recency half-life is 4 years, meaning a rating that is four years older than the user's newest rating receives approximately half as much recency weight.

### 2. Calculate the user's baseline rating

The user's average rating is computed using the recency weights.

This represents the user's normal rating level.

For example, a user who generally gives high ratings should not automatically appear to strongly prefer every genre.

### 3. Measure personal genre preference

For every candidate movie genre, the system looks at movies of that genre previously rated by the user.

Historical evidence is weighted by:

* rating recency
* similarity between the historical movie's release year and the candidate movie's release year

The weighted genre average is compared with the user's overall weighted average.

Conceptually:

`genre preference = weighted genre rating - weighted user average`

The result is also reduced when there is very little evidence for that genre.

The confidence calculation is:

`confidence = evidence / (evidence + confidence_prior)`

The current confidence prior is `5.0`.

This prevents one unusually high rating from creating an extremely strong genre preference.

### 4. Calculate movie quality

Movie quality is based on a Bayesian-adjusted average rating.

Movies with very few ratings are pulled toward the overall dataset average instead of trusting their raw average immediately.

Conceptually:

`quality_rating = (rating_count × movie_average + prior_strength × global_average) / (rating_count + prior_strength)`

The quality feature used by the recommender is:

`quality_score = quality_rating - global_average`

Therefore:

* positive quality score = above-average movie quality
* negative quality score = below-average movie quality

### 5. Produce the heuristic score

Before the learned reranker was introduced, the system estimated a movie rating using:

`predicted_rating = user_average + personal_score + quality_weight × quality_score`

The result is clipped to the MovieLens rating range of `0.5` to `5.0`.

The current heuristic quality weight is `0.20`.

### 6. Diversity reranking for the heuristic recommender

The heuristic recommender initially sorts movies by predicted rating and considers its top 50 candidates.

It then greedily selects recommendations while applying a penalty when a candidate repeats genres already present in selected recommendations.

The current repetition penalty is `0.08` per overlapping genre.

This prevents the recommendation list from becoming unnecessarily repetitive.

This diversity step is currently part of `recommend_by_genre`.

The current `recommend_with_ml` path ranks movies by ML score and does not yet apply the diversity reranker afterward.

### 7. Calculate popularity

Popularity is defined as:

`popularity = ln(1 + number of ratings)`

The logarithm prevents extremely popular movies from dominating the feature scale simply because they have many more ratings.

### 8. Create chronological training and evaluation partitions

For each eligible user, historical ratings are ordered chronologically and divided into:

* First 60%: user profile
* Next 20%: pairwise training movies
* Final 20%: held-out evaluation movies

The split is deterministic.

If multiple ratings share the same timestamp, movie ID is used as a deterministic tie-breaker.

Split fractions are validated so invalid configurations cannot silently produce incorrect partitions.

The profile, training, and test partitions are required to:

* contain no overlapping rows
* collectively cover the user's complete rating history

### 9. Build a leakage-safe global reference dataset

Before pairwise training examples are generated, all eligible users are chronologically split.

A global reference dataset is constructed using only profile and pairwise-training portions.

Held-out evaluation ratings are excluded from this dataset for every eligible user.

Therefore, population-level features such as movie quality and popularity cannot access any held-out evaluation ratings.

When constructing examples for a particular user, that user's own ratings are additionally removed from the population reference dataset.

This prevents the user's pairwise-training labels from influencing the population-level features used to predict those same preferences.

The information boundary is therefore:

`profile + training data → features/models`

`held-out test data → evaluation labels only`

### 10. Create pairwise training examples

The profile portion is used to calculate the user's preferences.

Movies in the pairwise-training portion are scored using only information available inside the training boundary.

Pairs of differently rated movies are then created.

For a preferred movie P and less-preferred movie L:

`X = features(P) - features(L), label = 1`

The reverse example is also included:

`X = features(L) - features(P), label = 0`

This creates a balanced pairwise classification dataset.

Pairs with equal ratings are skipped because there is no preference direction to learn.

### 11. Standardize the features

Before training, the feature matrix is standardized using `StandardScaler`.

This prevents features with naturally larger numerical ranges from receiving extra influence simply because of their scale.

The same fitted scaler is saved with the trained model and reused during inference.

### 12. Train logistic regression

A logistic regression classifier is trained on the pairwise feature differences.

The model learns how strongly changes in:

* personal preference
* quality
* popularity

correspond to user preference.

Training uses an explicit `random_state=42` to support reproducibility.

The trained scaler and logistic regression model are stored together as a `TrainedRanker`.

### 13. Rank unseen movies

During inference:

1. Movies already rated by the user are removed.
2. Every unseen movie receives personal, quality, and popularity features.
3. Features are transformed using the saved scaler.
4. Logistic regression's decision function produces an ML ranking score.
5. Movies are sorted by ML score from highest to lowest.
6. The top requested movies are returned.

## Features / Signals

### Personal Score

Measures whether the user historically rates the candidate's genres above or below their own normal rating level.

Affected by:

* genre overlap
* rating recency
* release-year similarity
* amount of historical evidence

### Quality Score

Measures how much the movie's Bayesian-adjusted rating differs from the global MovieLens average.

This provides a general movie-quality signal independent of a particular user's tastes.

### Popularity

Defined as:

`ln(1 + rating count)`

This gives the model information about how widely rated a movie is without allowing raw rating counts to grow excessively large.

## Training

The current reranker uses pairwise logistic regression.

Users with fewer than 10 ratings are skipped for model training because there is not enough history for the chronological split to provide useful profile and training portions.

Current leakage-safe training run:

* Users used: 605
* Pairwise training examples: 1,530,814
* Positive examples: 765,407
* Negative examples: 765,407

The model is trained using only information available inside the global profile + training boundary.

Held-out evaluation ratings are excluded from population-level feature construction.

The pairwise dataset is exactly balanced because every positive feature-difference example is paired with its reversed negative example.

## Reproducibility

Week 2 introduced explicit reproducibility safeguards.

These include:

* deterministic chronological splitting
* deterministic movie-ID tie-breaking for equal timestamps
* validated split fractions
* explicit `random_state=42` for logistic regression
* deterministic matrix-factorization initialization
* regression tests for information-boundary invariants

Two complete retraining and evaluation runs produced identical:

* training-user counts
* training-example counts
* test-pair counts
* model evaluation metrics
* confidence intervals
* diagnostic results

This verifies that the current training and evaluation pipeline is reproducible under the tested environment.

## Inference

The main inference entry point is:

`recommend_with_ml(...)`

It:

1. Loads the saved ranker.
2. Finds the user's watched movies.
3. Generates all unseen candidates.
4. Computes personal and quality signals through the genre recommender.
5. Computes movie popularity.
6. Standardizes the three features.
7. Calculates the logistic-regression decision score.
8. Sorts candidates by ML score.
9. Returns the top recommendations.

## Evaluation

The system uses chronological holdout evaluation rather than randomly mixing old and new ratings.

For each eligible user:

* 60% of historical ratings build the profile.
* 20% are used during model training.
* 20% remain held out for evaluation.

For the held-out portion, pairs of movies with different actual ratings are compared.

A ranking receives:

* `1.0` credit if it orders the pair correctly
* `0.0` credit if it orders the pair incorrectly
* `0.5` credit if the model predicts a tie

The evaluation framework compares the ML reranker against:

* a Bayesian movie-average baseline
* the handcrafted heuristic recommender
* biased matrix factorization

### Current Week 2 Leakage-Safe Benchmark

Users evaluated: 601

Test pairs: 695,795

Macro accuracy:

* Movie-average baseline: 62.730%
* Heuristic: 60.342%
* Matrix factorization: 62.773%
* ML reranker: 63.686%

ML improvement:

* vs movie-average baseline: +0.956 percentage points
* vs heuristic: +3.344 percentage points
* vs matrix factorization: +0.914 percentage points

Pair-weighted accuracy:

* Movie-average baseline: 67.084%
* Heuristic: 64.965%
* Matrix factorization: 66.964%
* ML reranker: 68.282%

Pair-weighted ML improvement:

* vs movie-average baseline: +1.198 percentage points
* vs heuristic: +3.317 percentage points
* vs matrix factorization: +1.318 percentage points

Per-user ML vs baseline results:

* ML better: 282 users (46.9%)
* Baseline better: 223 users (37.1%)
* Tied: 96 users (16.0%)

Per-user ML vs heuristic results:

* ML better: 343 users (57.1%)
* Heuristic better: 193 users (32.1%)
* Tied: 65 users (10.8%)

User-level bootstrap analysis:

* ML minus baseline mean difference: +0.956 percentage points
* 95% confidence interval: [+0.026%, +1.865%]

ML minus heuristic:

* Mean difference: +3.344 percentage points
* 95% confidence interval: [+2.215%, +4.471%]

The confidence interval for ML minus baseline remains slightly above zero, although the margin is modest.

A random ordering would be expected to perform around 50% on pairwise comparisons.

The current ML reranker performs above random, substantially improves on the handcrafted heuristic, and modestly outperforms both the movie-average baseline and matrix factorization when those models are evaluated independently.

## Evaluation by User History

Macro accuracy by total user rating count:

| Ratings | Users | Baseline | Heuristic | ML | ML − Baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10–29 | 102 | 63.168% | 56.489% | 65.347% | +2.179% |
| 30–49 | 114 | 61.803% | 58.150% | 61.820% | +0.017% |
| 50–99 | 137 | 61.714% | 61.060% | 62.583% | +0.870% |
| 100–199 | 114 | 62.849% | 61.448% | 63.531% | +0.682% |
| 200+ | 134 | 64.124% | 63.465% | 65.269% | +1.145% |

The reranker performs especially well relative to the baseline for users with very small histories and still maintains an advantage for users with large histories.

## Evaluation Sanity Checks

The evaluation pipeline verifies that:

* the number of stored user results matches the reported number of evaluated users
* per-user pair counts sum to the total pair count
* every evaluated user contributes at least one valid pair
* every evaluated user has at least 10 ratings
* baseline accuracy remains within `[0, 1]`
* heuristic accuracy remains within `[0, 1]`
* ML accuracy remains within `[0, 1]`
* matrix-factorization accuracy remains within `[0, 1]`

All current sanity checks pass.

## Reliability Tests

Week 2 added automated tests protecting the chronological and information-boundary logic.

Tests verify:

* chronological ordering
* deterministic behavior
* deterministic equal-timestamp tie-breaking
* no overlap between profile, training, and test partitions
* complete coverage of every rating
* rejection of invalid split fractions
* exclusion of held-out test ratings from global reference data
* exclusion of the target user's ratings from that user's population reference features

Current full automated test suite:

`58 passed`

## Important Hyperparameters

Current genre and heuristic parameters:

* Recency half-life: `4.0 years`
* Release-year penalty: `0.005 per year`
* Minimum release-year weight: `0.50`
* Genre confidence prior: `5.0`
* Heuristic quality weight: `0.20`
* Diversity repetition penalty: `0.08`
* Heuristic diversity candidate pool: `50 movies`

Movie-quality Bayesian prior strength:

* `10.0`

Pairwise training split:

* Profile: `60%`
* Training: `20%`
* Evaluation: `20%`

Popularity transformation:

* `ln(1 + rating_count)`

Logistic-regression random seed:

* `42`

Matrix-factorization random seed:

* `42`

## Known Limitations

* The ML model currently uses only three features.
* Personalization is primarily genre-based rather than learning deeper relationships between individual movies.
* Matrix-factorization predictions are currently evaluated separately rather than being fed into the ML reranker.
* Popularity can still introduce bias toward widely rated movies.
* The final ML recommendation path does not currently apply the diversity reranker.
* The model is trained globally rather than training a separate ranker for every individual user.
* MovieLens metadata is limited compared with production movie data.
* Release-year similarity is represented using a manually chosen linear penalty.
* Genre relationships are treated explicitly rather than learned automatically.
* MovieLens latest-small is useful for rapid iteration but limited in scale.
* Pairwise accuracy measures relative preference ordering and does not capture every aspect of recommendation quality.
* Offline MovieLens evaluation cannot fully represent real-world viewing intent or user satisfaction.

## Future Improvements

Possible improvements include:

* combine content-model features with collaborative-filtering predictions
* feed matrix-factorization scores directly into the hybrid reranker
* add director, actor, language, runtime, keyword, and embedding-based features
* add learned movie embeddings
* experiment with gradient-boosted or neural ranking models
* apply diversity reranking after the ML ranking stage
* optimize hyperparameters using validation data
* evaluate additional ranking metrics such as NDCG, Precision@K, Recall@K, and Hit Rate
* improve cold-start behavior for users with little rating history
* add calibrated predicted-enjoyment scores in addition to ranking scores
* increase dataset scale beyond MovieLens latest-small
* investigate group recommendation and multi-user preference aggregation
* investigate short-term viewing intent in addition to long-term user taste
* evaluate the system with real users after the offline recommender is stable

## Relevant Files

Primary implementation:

* `src/hybrid/genre_recommender.py`
* `src/hybrid/ml_reranker.py`

Training:

* `src/scripts/train_ml_reranker.py`

Saved trained model:

* `src/models/ml_reranker.joblib`

Evaluation infrastructure:

* `src/evaluation/ranking.py`
* `src/evaluation/splits.py`
* `src/evaluation/metrics.py`
* `src/evaluation/reports.py`

Hybrid tests:

* `tests/hybrid/test_genre_recommender.py`
* `tests/hybrid/test_ml_reranker.py`

Evaluation tests:

* `tests/evaluation/test_ranking.py`

Current demonstration entry point:

* `main.py`

---

# Benchmark History

Dataset: MovieLens latest-small

Evaluation split:

* 60% profile
* 20% pairwise training
* 20% held-out testing

## Week 2 Leakage-Safe and Reproducible Benchmark

Users evaluated: 601

Test pairs: 695,795

Macro results:

* Movie-average baseline accuracy: 62.730%
* Heuristic accuracy: 60.342%
* Matrix-factorization accuracy: 62.773%
* ML reranker accuracy: 63.686%
* ML improvement over heuristic: +3.344 percentage points
* ML improvement over movie-average baseline: +0.956 percentage points
* ML improvement over matrix factorization: +0.914 percentage points

Pair-weighted results:

* Movie-average baseline: 67.084%
* Heuristic: 64.965%
* Matrix factorization: 66.964%
* ML reranker: 68.282%

Training run:

* Users used: 605
* Pairwise training examples: 1,530,814
* Positive examples: 765,407
* Negative examples: 765,407

Reproducibility:

Two complete train-and-evaluate runs produced identical training counts, test-pair counts, accuracies, confidence intervals, and diagnostics.

Reliability changes introduced in Week 2:

* deterministic chronological splitting
* explicit validation of split fractions
* deterministic tie-breaking for equal timestamps
* tests proving complete split coverage
* tests proving no overlap between profile, training, and test partitions
* global held-out boundary excluding eligible users' evaluation ratings from population-level features
* target-user exclusion from population reference features
* automated leakage-regression tests
* explicit logistic-regression random seed
* retraining under the stronger information boundary
* reproducibility verification through repeated full training/evaluation runs
* evaluation sanity checks
* pair-weighted reporting
* per-user analysis
* bootstrap confidence intervals
* correlation diagnostics

## Earlier Week 2 Intermediate Benchmark

An intermediate Week 2 run produced:

* Users evaluated: 601
* Test pairs: 747,790
* Baseline accuracy: 62.630%
* Heuristic accuracy: 60.625%
* Matrix-factorization accuracy: 62.743%
* ML accuracy: 64.024%

This is not the final Week 2 benchmark.

Subsequent reliability changes altered the valid evaluation-pair set. After the pipeline was finalized, two complete retraining and evaluation runs produced the identical 695,795-pair benchmark documented above.

## Frozen Week 1 Benchmark

Users evaluated: 601

Test pairs: 747,939

Results:

* Movie-average baseline accuracy: 62.792%
* Heuristic accuracy: 60.597%
* ML reranker accuracy: 63.926%
* ML improvement over heuristic: +3.329 percentage points
* ML improvement over movie-average baseline: +1.134 percentage points

Training run:

* Users used: 604
* Pairwise training examples: 1,609,532
* Positive examples: 804,766
* Negative examples: 804,766

The Week 1 benchmark removed direct target-user leakage.

Population-level movie quality and popularity statistics excluded the target user's own ratings when generating that user's features.

During the Week 2 reliability audit, an additional information-boundary issue was discovered: population-level features could still use held-out ratings belonging to other users.

Week 2 introduced a stronger global chronological boundary in which held-out evaluation ratings from eligible users are excluded from population-level feature construction.

The Week 1 benchmark remains frozen as a historical benchmark.

## Superseded Initial Benchmark

An earlier evaluation produced:

* Heuristic accuracy: 61.995%
* ML reranker accuracy: 66.950%
* Improvement: +4.955 percentage points
* Test pairs: 941,248

This result is not an official benchmark.

The initial implementation calculated population-level movie quality and popularity using the complete ratings dataset.

As a result, hidden ratings belonging to the user being evaluated could influence features used to evaluate that same user.

The evaluation pipeline was subsequently corrected, strengthened during the Week 2 reliability audit, and the model was retrained.