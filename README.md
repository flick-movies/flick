# Flick

Flick is a personalized movie recommendation system designed to learn what individual users enjoy and rank movies accordingly.

The project is currently focused on building and evaluating the recommendation engine that will power the future Flick application.

## Recommendation System

Flick's recommendation architecture is being developed around three components:

### Content-Based Recommendation

Models a user's preferences from properties of movies they have previously rated, such as genres and other movie metadata.

### Collaborative Filtering

Learns from rating patterns across users to identify movies a user may enjoy based on the behavior of users with similar preferences.

### Hybrid / Reranking

Combines recommendation signals and reranks candidate movies to produce the final personalized ranking.

The current hybrid system uses a pairwise logistic regression reranker with three features:

- Personal preference score
- Movie quality score
- Movie popularity

Future versions will incorporate outputs from both the content-based and collaborative models into a unified hybrid ranking system.

## Current Benchmark

**Dataset:** MovieLens 1M

### Evaluation Protocol

For each eligible user, ratings are ordered chronologically and divided into:

- First 60% → preference profile
- Next 20% → pairwise training
- Final 20% → held-out testing

Population-level quality and popularity statistics exclude the target user's ratings during training and evaluation to prevent the user's hidden ratings from influencing features used to predict that same user's preferences.

### Week 1 Frozen Benchmark

| Model | Pairwise Accuracy |
| --- | ---: |
| Movie-average baseline | 62.792% |
| Handcrafted heuristic | 60.597% |
| **ML reranker** | **63.926%** |

The ML reranker improves by:

- **+3.329 percentage points** over the handcrafted heuristic
- **+1.134 percentage points** over the movie-average baseline

Evaluation included:

- **601 users**
- **747,939 held-out rating pairs**
- Identical results across repeated evaluation runs
- Full automated test suite passing before the benchmark was frozen

The benchmark is frozen under the Git tag:

`week1-reranker-benchmark`

More detailed methodology and architecture are documented in `docs/hybrid-reranking.md`.

## Current Project Status

Flick is currently in the recommender-system development stage.

Current work includes:

- Content-based recommendation
- Collaborative filtering
- Learned hybrid reranking
- Leakage-safe offline evaluation
- Recommendation-system testing
- Architecture and integration planning

Planned development includes:

- Integration of content and collaborative model outputs
- Stronger hybrid ranking
- Recommendation quality analysis and evaluation
- Group recommendations
- Diversity and fairness-aware reranking
- Application and backend integration

## Repository Structure

```text
src/
├── content/          # Content-based recommendation
├── collaborative/    # Collaborative filtering
├── hybrid/           # Hybrid models and reranking
├── evaluation/       # Offline recommender evaluation
├── data_processing/  # Dataset processing
└── models/           # Saved model artifacts

tests/
├── content/
├── collaborative/
└── hybrid/

docs/                 # Architecture and model documentation
```

## Dataset

Current development and evaluation use the MovieLens 1M dataset.

MovieLens provides approximately one million movie ratings from thousands of users and is used as an offline benchmark while Flick's recommendation system is developed.

## Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest
```

Run the current recommendation demo:

```bash
python main.py
```

## Project Goal

Flick's goal is to move beyond generic movie popularity and produce recommendations that reflect each user's individual taste.

The current system establishes the first reproducible baseline for measuring that goal. Future model changes will be evaluated against frozen benchmarks rather than judged only by whether individual recommendations appear reasonable.
