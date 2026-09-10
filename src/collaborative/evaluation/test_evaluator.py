from pathlib import Path
import sys
import argparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from src.collaborative.evaluation.evaluator import evaluate_models, confidence_report, sparse_report


def main():
    parser = argparse.ArgumentParser(description="Collaborative chronological evaluation")
    parser.add_argument("--split", choices=["chronological", "random"], default="chronological")
    args = parser.parse_args()
    ratings = pd.read_csv(Path(__file__).resolve().parents[3] / "data" / "ratings.csv")
    if args.split == "chronological":
        print("Collaborative chronological 60/20/20: fit profile + train; evaluate held-out test.", flush=True)
        print("Standalone rating/confidence diagnostics, not the hybrid team ranking benchmark.")
    else:
        print("DIAGNOSTIC ONLY: random 80/20, seed 42. Not the team benchmark.", flush=True)
    results, predictions = evaluate_models(ratings, return_details=True, split=args.split)
    print("\nBaseline vs MF (lower RMSE/MAE is better):")
    print(pd.DataFrame(results).T.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nMF confidence analysis (accuracy is a fraction; NaN means no observations):")
    print(confidence_report(predictions).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSparse evidence analysis (groups overlap):")
    print(sparse_report(predictions).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nConfidence is an evidence heuristic, not probability of correctness. Compare error and bucket counts.")


if __name__ == "__main__":
    main()
