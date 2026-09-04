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

### 8. Create pairwise training examples

For each eligible user, historical ratings are divided chronologically:

* First 60%: user profile
* Next 20%: pairwise training movies
* Final 20%: held-out evaluation movies

The profile portion is used to calculate the user's preferences.

Movies in the training portion are scored using only the earlier profile.

Pairs of differently rated training movies are then created.

For a preferred movie P and less-preferred movie L:

`X = features(P) - features(L), label = 1`

The reverse example is also included:

`X = features(L) - features(P), label = 0`

This creates a balanced pairwise classification dataset.

Pairs with equal ratings are skipped because there is no preference direction to learn.

Before pairwise examples are generated, all eligible users are split chronologically.

A global reference dataset is constructed using only profile and pairwise-training portions. Held-out evaluation ratings are excluded from this dataset for every eligible user.

Population-level features such as movie quality and popularity are therefore computed without access to any held-out evaluation ratings.

When constructing training examples for a particular user, that user's ratings are removed from the population reference dataset so that their pairwise-training labels cannot influence the population-level features used to predict those same preferences.

### 9. Standardize the features

Before training, the feature matrix is standardized using `StandardScaler`.

This prevents features with naturally larger numerical ranges from receiving extra influence simply because of their scale.

The same fitted scaler is saved with the trained model and reused during inference.

### 10. Train logistic regression

A logistic regression classifier is trained on the pairwise feature differences.

The model learns coefficients indicating how strongly changes in:

* personal preference
* quality
* popularity

correspond to user preference.

The trained scaler and logistic regression model are stored together as a `TrainedRanker`.

### 11. Rank unseen movies

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

The model is trained using only information available inside the global profile + training boundary. Held-out evaluation ratings are excluded from population-level feature construction.

Because the features are standardized before training, learned logistic-regression coefficients can be compared more meaningfully than coefficients learned from raw feature scales.

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

Current Week 2 leakage-safe evaluation:

* Users evaluated: 601
* Test pairs: 747,790
* Movie-average baseline accuracy: 62.630%
* Heuristic pairwise accuracy: 60.625%
* Matrix-factorization accuracy: 62.743%
* ML reranker accuracy: 64.024%

ML improvement:

* vs movie-average baseline: +1.393 percentage points
* vs heuristic: +3.399 percentage points
* vs matrix factorization: +1.280 percentage points

Pair-weighted accuracy:

* Movie-average baseline: 66.698%
* Heuristic: 65.582%
* Matrix factorization: 66.716%
* ML reranker: 69.203%

User-level bootstrap analysis:

* ML minus baseline mean difference: +1.393 percentage points
* 95% confidence interval: [+0.371, +2.406] percentage points

A random ordering would be expected to perform around 50% on pairwise comparisons.

The current ML reranker performs meaningfully above random, improves on the handcrafted heuristic, and modestly outperforms both the movie-average baseline and matrix factorization when evaluated independently.

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

## Known Limitations

* The ML model currently uses only three features.
* Personalization is primarily genre-based rather than learning deeper relationships between individual movies.
* Matrix-factorization predictions are currently evaluated separately rather than being fed into the ML reranker.
* Popularity can still introduce some bias toward widely rated movies.
* The final ML recommendation path does not currently apply the diversity reranker.
* The model is trained globally rather than training a separate ranker for every individual user.
* MovieLens metadata is limited compared with production movie data.
* Release-year similarity is represented using a manually chosen linear penalty.
* Genre relationships are treated explicitly rather than learned automatically.
* The current evaluation dataset is MovieLens latest-small, which is useful for iteration but limited in scale.

## Future Improvements

Possible improvements include:

* Combine content-model features with collaborative-filtering predictions.
* Feed matrix-factorization scores directly into the hybrid reranker.
* Add director, actor, language, runtime, keyword, and embedding-based features.
* Add learned movie embeddings.
* Experiment with gradient-boosted or neural ranking models.
* Apply diversity reranking after the ML ranking stage.
* Optimize hyperparameters using validation data.
* Evaluate additional ranking metrics such as NDCG, Precision@K, Recall@K, and Hit Rate.
* Improve cold-start behavior for users with little rating history.
* Add calibrated predicted enjoyment scores in addition to ranking scores.
* Increase dataset scale beyond MovieLens latest-small.
* Investigate group recommendation and multi-user preference aggregation.
* Investigate short-term viewing intent in addition to long-term user taste.

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

## Week 2 Leakage-Safe Benchmark

Users evaluated: 601  
Test pairs: 747,790

Results:

* Movie-average baseline accuracy: 62.630%
* Heuristic accuracy: 60.625%
* Matrix-factorization accuracy: 62.743%
* ML reranker accuracy: 64.024%
* ML improvement over heuristic: +3.399 percentage points
* ML improvement over movie-average baseline: +1.393 percentage points
* ML improvement over matrix factorization: +1.280 percentage points

Pair-weighted results:

* Movie-average baseline: 66.698%
* Heuristic: 65.582%
* Matrix factorization: 66.716%
* ML reranker: 69.203%

Training run:

* Users used: 605
* Pairwise training examples: 1,530,814
* Positive examples: 765,407
* Negative examples: 765,407

Reliability changes introduced in Week 2:

* deterministic chronological splitting
* explicit validation of split fractions
* deterministic tie-breaking for equal timestamps
* tests proving complete split coverage
* tests proving no overlap between profile, training, and test partitions
* global held-out boundary excluding all eligible users' evaluation ratings from population-level features
* target-user exclusion from population reference features
* retraining under the stronger information boundary
* evaluation sanity checks
* pair-weighted reporting
* bootstrap confidence intervals

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

Week 2 introduced a stronger global chronological boundary in which held-out evaluation ratings from all eligible users are excluded from population-level feature construction.

The Week 1 benchmark remains frozen as a historical benchmark.

## Superseded Initial Benchmark

An earlier evaluation produced:

* Heuristic accuracy: 61.995%
* ML reranker accuracy: 66.950%
* Improvement: +4.955 percentage points
* Test pairs: 941,248

This result is not an official benchmark.

The initial implementation calculated population-level movie quality and popularity using the complete ratings dataset. As a result, hidden ratings belonging to the user being evaluated could influence features used to evaluate that same user.

The evaluation pipeline was subsequently corrected and the model was retrained.