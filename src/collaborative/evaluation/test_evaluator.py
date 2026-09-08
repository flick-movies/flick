from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from src.collaborative.evaluation.evaluator import evaluate_models, confidence_report, sparse_report


def main():
    ratings = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "ratings.csv")
    print("Collaborative evaluation: random 80/20 holdout, seed 42; training-only evidence.", flush=True)
    results, predictions = evaluate_models(ratings, return_details=True)
    print("\nBaseline vs MF (lower RMSE/MAE is better):")
    print(pd.DataFrame(results).T.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMF confidence analysis (accuracy is a fraction; NaN means no observations):")
    print(confidence_report(predictions).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSparse evidence analysis (groups overlap):")
    print(sparse_report(predictions).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nConfidence is an evidence heuristic, not probability of correctness. Compare error and bucket counts.")


if __name__ == "__main__":
    main()
