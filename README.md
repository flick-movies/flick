# Penumbra

Penumbra is a personalized movie recommendation system designed to learn what individual users enjoy and produce ranked movie recommendations tailored to their taste.

The project focuses on building an understandable, testable, and modular recommendation engine that combines multiple sources of evidence about users and movies.

## Recommendation System

Penumbra's recommendation architecture is built around three cooperating components:

### Content-Based Recommendation

Models a user's preferences from properties of movies they have previously rated.

The content system learns a user taste profile from movie metadata and produces personalized movie predictions together with confidence and reason signals.

Current content features include signals such as:

- Genre
- Director
- Runtime
- Release era
- Language
- User rating history

### Collaborative Filtering

Learns from rating behavior across users to identify movies a user may enjoy based on population-level preference patterns.

The collaborative system includes matrix factorization, which learns latent representations of users and movies from historical ratings. It also handles sparse data and produces confidence estimates based on available evidence.

### Hybrid / Reranking

Combines recommendation signals into a final personalized ranking.

The hybrid system uses learned pairwise ranking to determine which movies should appear above others for a user. It can incorporate signals from:

- Content-based predictions
- Collaborative predictions
- Personal preference scores
- Movie quality
- Movie popularity

Hybrid architectures are evaluated against standalone models and frozen baselines before being adopted.

Detailed methodology, experiments, benchmarks, and ablation results are documented in `docs/hybrid-reranking.md`.

## Architecture

The recommendation system follows the general pipeline:

```text
                    ┌─────────────────┐
User Ratings ──────►│ Content Model   │──────┐
                    └─────────────────┘      │
                                             ▼
                                      ┌───────────────┐
                                      │ Hybrid        │
                                      │ Reranker      │──► Ranked Movies
                                      └───────────────┘
                                             ▲
                    ┌─────────────────┐      │
Rating Data ───────►│ Collaborative   │──────┘
                    │ Model           │
                    └─────────────────┘