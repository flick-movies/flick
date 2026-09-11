# Collaborative Filtering

## Purpose

The collaborative recommendation system predicts how much a user will enjoy a
movie by learning from rating patterns across users and movies.

The component contains two models:

1. A movie-average baseline that estimates general movie quality.
2. Biased matrix factorization that learns personalized user/movie relationships.

Role 2 supplies predicted ratings and confidence. Candidate selection, watched-movie
filtering and final individual/group ranking belong to the hybrid component.

## Inputs

Both models use MovieLens rating records from `data/ratings.csv`.

Required training columns:

* `movieId` and `rating` for the baseline.
* `userId`, `movieId` and `rating` for matrix factorization.

Prediction takes matching sequences of user IDs and movie IDs. Each position
represents one user/movie pair. Movie titles, genres and other metadata are not
needed. Timestamps are present in the dataset but are not used by the current
random-split collaborative evaluator.

## Outputs

Both models return a pandas DataFrame containing:

* `user_id`
* `movie_id`
* `predicted_score`
* `confidence`

`predicted_score` estimates a star rating. MF clips scores to 0.5-5.0;
the baseline clips to 0.0-5.0. Confidence is an evidence heuristic between 0 and 1,
not a probability that the prediction is correct. Empty batches retain these columns.

## Core Idea

The baseline predicts the same movie score for every user. It shrinks movies with
few ratings toward the global training average:

```
baseline_score = (movie_count * movie_average + prior_strength * global_average)
                 / (movie_count + prior_strength)
```

Matrix factorization represents every known user and movie with learned numeric
vectors. Their dot product describes a personalized match beyond general rating
tendencies. The vector dimensions are learned patterns, not predefined genres.

The Week 1 prediction rule is:

```
score = global_average + user_bias + movie_bias + dot(user_factors, movie_factors)
```

The current Week 2 default reduces weakly supported latent interactions:

```
score = global_average + user_bias + movie_bias
        + confidence * dot(user_factors, movie_factors)
```

`shrink_latent=False` reproduces the Week 1 inference rule. This changes inference
only; the evaluator compares both rules using identical trained weights.

## Algorithm Flow

### 1. Prepare observed ratings

Check required columns and drop rows missing required values. Empty training data
raises an error. Training uses observed rating rows directly; missing user/movie
ratings are not filled with zero and no dense ratings matrix is allocated.

### 2. Fit the movie-average baseline

Calculate the global average and each movie's average and rating count. The prior
strength determines how strongly a rare movie is pulled toward the global mean.

### 3. Map users and movies to array positions

Convert external IDs into consecutive indices. Record user and movie training
counts for confidence calculations and unknown-ID detection.

### 4. Initialize the personalized model

Set user and movie biases to zero. Initialize small random user/movie factor
vectors using the configured random seed.

### 5. Learn biases and taste vectors

Shuffle observed ratings each epoch. Predict each rating and calculate:

`error = actual_rating - predicted_rating`

Stochastic gradient descent updates the relevant user bias, movie bias and factor
vectors. The updates reduce squared error while regularization penalizes large
weights. The movie-factor update uses a copy of the user factors from before that
rating's update.

### 6. Calculate evidence and confidence

For a known user and movie:

```
user_evidence = user_count / (user_count + prior_strength)
movie_evidence = movie_count / (movie_count + prior_strength)
confidence = user_evidence * movie_evidence
```

Counts come only from the fitted training data. A weak user or movie history
reduces confidence. Week 2 uses this confidence to shrink the latent interaction
toward the regularized bias prediction. The baseline separately uses
`movie_count / (movie_count + prior_strength)` as its confidence.

### 7. Handle unknown IDs and return predictions

Unknown users/movies have no learned interaction. The model uses available biases
or the global mean, clips the result and returns the shared output columns.

## Features / Signals

### Global Average

The average training rating provides a default starting point.

### User Bias

Captures whether a user tends to rate more generously or harshly than average.

### Movie Bias

Captures whether a movie tends to receive higher or lower ratings than average.

### Latent Interaction

The dot product captures personalized rating patterns that the biases do not
explain. It can increase or decrease a predicted rating.

### Confidence

Summarizes the amount of available training evidence. Its usefulness is checked
by measuring held-out rating error in confidence buckets.

## Training

The evaluator trains both models on exactly the same random 80% of rating rows.
The remaining 20% are used only for evaluation. MF optimizes observed rating error
with regularization; it is not trained as a pairwise ranking classifier.

The fixed seed reproduces initialization, shuffling and splitting for the same
input order and environment. `ModelWeights` stores learned arrays, ID mappings,
training counts and the global average.

`save()` stores the fitted model and settings with joblib; `load()` restores them.
The default path is `src/models/collaborative_mf.joblib`. The benchmark trains in
memory and does not save an artifact. Older saved objects without `shrink_latent`
preserve their original inference behavior.

## Inference

The main interface is:

`model.predict(user_ids, movie_ids)`

It returns one row per paired request, preserving order. It does not generate
candidates, remove watched movies or return a final ranked recommendation list.

Unknown-ID behavior is explicit:

| User | Movie | Prediction before clipping | Confidence |
|---|---|---|---|
| Known | Known | Global + both biases + latent adjustment | User evidence * movie evidence |
| Known | Unknown | Global + user bias | 0.1 |
| Unknown | Known | Global + movie bias | 0.5 * movie evidence |
| Unknown | Unknown | Global average | 0.0 |

These fallback confidence values are heuristics. Tests cover all combinations,
bounded output, empty batches and sparse-evidence shrinkage.

## Evaluation

The dedicated evaluator is `src/collaborative/evaluation/evaluator.py`.
It uses `train_test_split(test_size=0.2, random_state=42)` and scores every held-out
row with both models, including movies absent from training. Model fitting,
movie averages, counts and confidence never use the test ratings.

Metrics:

* MAE: mean absolute rating error; lower is better.
* RMSE: square root of mean squared error; lower is better and large errors count more.
* Accuracy within 0.5: fraction of predictions within half a star; higher is better.

Confidence buckets include their lower boundary and exclude their upper boundary,
except the final bucket includes 1. Empty groups report count zero and NaN metrics.
Sparse-group results use training counts and can overlap.

`evaluate_models(ratings, return_details=True)` also returns actual ratings,
baseline/MF/original-MF predictions, confidence, absolute error and training counts.
The default return is a model-to-metrics dictionary.

From the repository root, run either:

```powershell
.venv/Scripts/python.exe -m src.collaborative.evaluation.test_evaluator
.venv/Scripts/python.exe src/collaborative/evaluation/test_evaluator.py
```

The confidence demo supports the same two launch styles. Direct script execution
resolves imports and dataset paths relative to the script, so it also works from
another working directory when absolute script/interpreter paths are supplied.
Module execution requires the repository root as the working directory.

Run all existing project tests with:

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests src/tests
```

## Important Hyperparameters

* Baseline prior strength: `10.0`.
* MF latent factors: `20`.
* Learning rate: `0.005`.
* Regularization: `0.02`.
* Training epochs: `20`.
* MF confidence prior strength: `5.0`, finite and positive.
* Random seed: `42`.
* Train/test split: `80% / 20%`.
* `shrink_latent`: `False` for the Week 1 benchmark; current default `True`.

## Known Limitations

* Confidence is not statistically calibrated.
* The random holdout does not measure future behavior under a chronological split.
* The hybrid document uses pairwise accuracy and a different split; its metrics
  cannot be compared numerically with collaborative rating-error metrics.
* This dataset split has no unknown users or users with at most five training
  ratings. Controlled tests verify behavior, but not real predictive accuracy,
  for those cases.
* The nested Python training loop can become slow on larger datasets.
* Hyperparameters have not been tuned against a separate validation set.
* The current hybrid recommendation path does not consume collaborative scores.
* The Week 2 safeguard has a small overall error regression; see results below.

## Future Improvements

* Tune on validation data and retain an untouched test set for final measurement.
* Compare models using the team's shared chronological split and ranking metrics.
* Measure cold-start accuracy on dedicated sparse-user scenarios.
* Integrate collaborative scores and confidence into the hybrid feature pipeline.
* Validate confidence calibration and optimize training/inference speed.

## Relevant Files

Primary implementation:

* `src/collaborative/baseline.py`
* `src/collaborative/matrix_factorization.py`

Evaluation and demos:

* `src/collaborative/evaluation/evaluator.py`
* `src/collaborative/evaluation/metrics.py`
* `src/collaborative/evaluation/test_evaluator.py`
* `src/collaborative/evaluation/test_mf_confidence.py`

Tests:

* `tests/collaborative/test_baseline.py`
* `tests/collaborative/test_matrix_factorization.py`
* `tests/collaborative/test_reliability.py`
* Legacy collaborative tests also remain in `src/tests/`.

# Week 1 Collaborative Benchmark

## Current Reproduction of the Week 1 Model

These results were rerun using the current local dataset and evaluator. They
reproduce the original unshrunk MF inference rule; they are not represented as
an archived historical run or a frozen team-approved benchmark.

Dataset: the bundled MovieLens small CSVs, containing 100,836 ratings.

* Training ratings: 80,668.
* Held-out ratings: 20,168.
* Split: random 80/20, seed 42.
* MF: default training settings above, `shrink_latent=False` at inference.
* Runtime: Python 3.12, NumPy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0,
  joblib 1.5.3 and SciPy 1.18.0.

| Model | RMSE | MAE | Accuracy within 0.5 |
|---|---:|---:|---:|
| Movie-average baseline | 0.9735 | 0.7568 | 42.06% |
| Week 1 matrix factorization | 0.8781 | 0.6707 | 47.48% |

MF has lower rating error than the movie-average baseline on the same held-out
ratings. These are rating predictions, so the accuracy column means half-star
tolerance, not pairwise ranking accuracy.

The evaluator prints this Week 1 row as `MF without shrinkage`.

## Week 2 Reliability Comparison

The same run also evaluates the current evidence-aware inference rule using the
same trained weights and held-out rows.

| Model | RMSE | MAE | Accuracy within 0.5 |
|---|---:|---:|---:|
| Week 1 MF | 0.8781 | 0.6707 | 47.48% |
| Week 2 MF | 0.8782 | 0.6708 | 47.49% |

The safeguard reduces weakly supported latent contributions. Overall MAE/RMSE
worsen by about 0.0001, so this is not an overall accuracy improvement. For 3,447
rare-movie rows (at most five training ratings), MAE changes from 0.7558 to 0.7556.

### Confidence Analysis for the Current Week 2 Model

| Confidence | Count | RMSE | MAE | Accuracy within 0.5 |
|---|---:|---:|---:|---:|
| [0.00, 0.20) | 1,448 | 1.0177 | 0.7782 | 41.51% |
| [0.20, 0.40) | 1,116 | 0.9777 | 0.7584 | 42.11% |
| [0.40, 0.60) | 1,961 | 0.9003 | 0.6956 | 45.95% |
| [0.60, 0.80) | 4,926 | 0.8749 | 0.6701 | 47.67% |
| [0.80, 1.00] | 10,717 | 0.8437 | 0.6430 | 49.06% |

Higher confidence corresponds to lower error in this split. This supports using
confidence as an evidence indicator, but does not establish calibrated probabilities
or statistical significance. The test set contains 813 unknown-movie rows and zero
unknown-user rows. The full automated suite passed: 54 tests, including direct-file
import regression checks for both collaborative demos.
