# Collaborative evaluation

Run from the repository root:

```
.venv/Scripts/python.exe -m src.collaborative.evaluation.test_evaluator
```

The command defaults to per-user chronological 60/20/20, ordered by timestamp
then movie ID. The first 60% is the profile and the next 20% is training;
both collaborative models fit those two portions together. The last 20% is
held out for MAE, RMSE, half-star accuracy and confidence diagnostics.
Users with fewer than 10 ratings contribute training rows only.
Counts and model statistics use training data only.

All implementation lives in this directory and uses only the collaborative
models. It does not call or modify the hybrid/shared evaluator. These are
standalone rating diagnostics, not the hybrid candidate-filtered ranking benchmark.

Use `--split random` for the historical random 80/20 diagnostic.
The Python `evaluate_models` API retains its old random default for existing
callers; pass `split="chronological"` to select the new method explicitly.
The command-line default is chronological.

Tests:

```
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests src/tests src/collaborative/evaluation/test_chronological.py
```
