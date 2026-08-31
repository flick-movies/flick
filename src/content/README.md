# Content Model

The content model learns a user's movie preferences from movie metadata. Its responsibility is to produce a predicted score from 0 to 5, confidence from 0 to 1, and structured reason signals for a user and movie pair.

The package does not own collaborative filtering, global popularity ranking, group aggregation, diversity reranking, candidate generation, or the final hybrid ranking decision.

## Current implementation

- `schemas.py` defines the typed input and output contract.
- `baselines.py` calculates each user's personal rating mean and rating residuals.
- `genres.py` normalizes multi-genre evidence and aggregates raw per-genre preferences.
- `profiles.py` builds a reusable, versioned `UserTasteProfile` with baseline, genre preferences, evidence counts, and metadata coverage statistics.
- `scoring.py` calculates a bounded genre component and a clamped personalized prediction.
- `model.py` caches profiles and provides single and deterministic batch prediction APIs.
- `errors.py` defines explicit unknown-user and unknown-movie errors.
- `demo.py` prints the profile, candidate features, genre component, and final score for real MovieLens records.
- `src/data_processing/movielens.py` converts MovieLens records into the content-model schemas.

## Profile building

Rating residuals are calculated as:

```text
residual = rating - user_mean
```

For a rated movie with `N` unique genres, its residual is divided by `N`. This prevents a movie with many genres from contributing more total learning signal than a movie with one genre. Missing genres contribute no genre evidence.

Each genre stores its total normalized contribution, mean normalized contribution, and rated-movie count. A profile also records how many ratings had movie metadata, genre metadata, missing movie metadata, and missing genres. Profiles use the version `genre-v1`.

## Prediction

For a candidate movie, each unique genre contributes the user's mean learned preference for that genre. Unknown genres contribute `0`. The component is the average across the candidate's genres and is bounded to `-1` through `+1` by default.

```text
raw_genre_component = average(candidate_genre_preferences)
bounded_genre_component = clamp(raw_genre_component, -1, 1)
weighted_adjustment = genre_weight * bounded_genre_component
predicted_score = clamp(user_baseline + weighted_adjustment, 0, 5)
```

The default genre weight is `1.0`. `ScoringConfig` makes both the weight and component bound explicit and reusable.

Clamping is only a safety boundary. It keeps the public score inside `0` to `5`; it does not mean the score is calibrated.

## Public API

```python
from src.content import ContentModel

model = ContentModel(ratings, movies)
profile = model.build_profile(user_id=1)
one_result = model.predict_one(user_id=1, movie_id=8, include_debug=True)
batch_results = model.predict(user_ids=(1, 2), movie_ids=(8, 9))
unseen_results = model.predict_unseen(user_ids=(1, 2), limit=10)
```

`ContentModel` caches each profile after its first build. Batch prediction returns the Cartesian product in user order and then movie order. For the example above, the order is `(1, 8)`, `(1, 9)`, `(2, 8)`, `(2, 9)`.

`predict_unseen` removes every movie the user has already rated, orders the remaining movie IDs deterministically, applies the optional limit separately for each user, and returns predictions in user order. More advanced candidate retrieval remains outside this package.

An unknown user raises `UnknownUserError`. An unknown movie ID raises `UnknownMovieError`. A known movie with no genres receives the user's baseline because its genre adjustment is zero.

When `include_debug=True`, each result includes:

- Baseline
- Candidate genres
- Matched and unknown genres
- Raw and bounded genre components
- Genre weight and weighted adjustment
- Unclamped score
- Whether clamping changed the score

The stable prediction contract is:

```text
user_id
movie_id
predicted_score
confidence
reason_signals
```

Confidence is deliberately `0.0` and reason signals are empty in this Week 1 implementation. Proper evidence-aware confidence and truthful reason-signal thresholds belong to the next model stage. Returning placeholders avoids claiming certainty or explanations that have not been implemented yet.

## Verification

Run all tests from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The deterministic suite covers schemas, exact baselines and residuals, genre normalization, profile construction, positive and negative predictions, neutral and missing genres, multi-genre scoring, configurable weighting, score clamping, cache reuse, batch order, unseen filtering, and unknown IDs.

Inspect a real MovieLens user's profile and prediction calculations:

```bash
python3 -m src.content.demo
```
