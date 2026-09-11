import numpy as np
import pandas as pd
import pytest

from src.collaborative.matrix_factorization import BiasedMatrixFactorization
from src.collaborative.evaluation.evaluator import confidence_report, evaluate_models
from src.collaborative.evaluation.metrics import mae, rmse


def fitted():
    return BiasedMatrixFactorization(n_epochs=2).fit(pd.DataFrame({
        "userId": [1, 1, 2, 2, 2], "movieId": [10, 20, 10, 20, 30],
        "rating": [5., 3., 4., 2., 4.]}))


def test_sparse_latent_adjustment_and_confidence():
    model = fitted()
    w = model.weights_
    # A controlled positive interaction lets us isolate inference from training.
    w.user_factors[:] = 0.1
    w.movie_factors[:] = 0.1
    w.user_biases[:] = 0
    w.movie_biases[:] = 0
    sparse = model.predict([1], [30]).iloc[0]
    supported = model.predict([2], [10]).iloc[0]
    assert 0 < sparse.confidence < supported.confidence < 1
    model.shrink_latent = False
    original = model.predict([1], [30]).iloc[0]
    assert w.global_mean < sparse.predicted_score < original.predicted_score


def test_unknown_fallbacks_and_batch_contract():
    model = fitted()
    w = model.weights_
    result = model.predict([1, 999, 1, 999], [10, 10, 999, 999])
    assert list(result.columns) == ["user_id", "movie_id", "predicted_score", "confidence"]
    assert np.isfinite(result.to_numpy()).all()
    assert result.predicted_score.between(.5, 5).all()
    assert result.confidence.between(0, 1).all()
    assert result.iloc[1].predicted_score == pytest.approx(w.global_mean + w.movie_biases[w.movie_to_idx[10]])
    assert result.iloc[2].predicted_score == pytest.approx(w.global_mean + w.user_biases[w.user_to_idx[1]])
    assert result.iloc[3].predicted_score == pytest.approx(w.global_mean)
    assert result.iloc[3].confidence == 0
    assert model.predict([], []).empty


@pytest.mark.parametrize("prior", [0, -1, np.nan, np.inf])
def test_invalid_confidence_prior(prior):
    with pytest.raises(ValueError):
        BiasedMatrixFactorization(prior_strength=prior)


def test_metrics_and_bucket_boundaries():
    assert mae([1, 3], [2, 1]) == 1.5
    assert rmse([1, 3], [2, 1]) == pytest.approx(np.sqrt(2.5))
    predictions = pd.DataFrame({"rating": [3.] * 6, "mf_score": [2.] * 6,
                                "confidence": [0, .2, .4, .6, .8, 1]})
    report = confidence_report(predictions)
    assert report.Count.tolist() == [1, 1, 1, 1, 2]
    assert (report.MAE == 1).all()
    empty = confidence_report(predictions.iloc[:1])
    assert empty.iloc[1].Count == 0
    assert np.isnan(empty.iloc[1].MAE)


def test_evaluation_uses_training_only_evidence():
    # Every movie appears once: every held-out movie MUST be unknown.
    ratings = pd.DataFrame({"userId": [1] * 20, "movieId": range(20), "rating": [3.] * 20})
    results, predictions = evaluate_models(ratings, return_details=True)
    assert len(predictions) == 4
    assert (predictions.movie_count == 0).all()
    assert (predictions.user_count == 16).all()
    assert (predictions.baseline_score == 3).all()
    assert results["Baseline"]["RMSE"] == 0
    assert all(set(values) == {"RMSE", "MAE", "Accuracy within 0.5"} for values in results.values())


@pytest.mark.parametrize("script", ["test_evaluator.py", "test_mf_confidence.py"])
def test_demo_imports_from_outside_repository(script, tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "collaborative" / "evaluation" / script
    # A non-main run loads each entry point without starting expensive training.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import runpy, sys; runpy.run_path(sys.argv[1])", str(path)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
