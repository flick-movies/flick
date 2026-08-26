# Content Model

The content model learns a user's movie preferences from movie metadata. Its responsibility is to produce a predicted score from 0 to 5, confidence from 0 to 1, and structured reason signals for a user and movie pair.

The package does not own collaborative filtering, global popularity ranking, group aggregation, diversity reranking, or the final hybrid ranking decision.

## Current foundation

- `schemas.py` defines the typed input and output contract.
- `baselines.py` calculates each user's personal rating mean and rating residuals.
- `genres.py` normalizes multi-genre evidence and aggregates raw per-genre preferences.
- `src/data_processing/movielens.py` converts MovieLens records into the content-model schemas.

Rating residuals are calculated as:

```text
residual = rating - user_mean
```

For a movie with `N` unique genres, its residual is divided by `N`. This prevents a movie with many genres from contributing more total learning signal than a movie with one genre. Missing genres contribute no genre evidence.

The stable prediction contract is:

```text
user_id
movie_id
predicted_score
confidence
reason_signals
```

Run the foundation tests from the repository root:

```text
python -m unittest discover -s tests -v
```

Inspect a real MovieLens user's baseline and raw genre preference table:

```text
python -m src.content.demo
```
