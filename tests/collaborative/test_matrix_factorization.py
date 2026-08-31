import pandas as pd
from src.collaborative.matrix_factorization import BiasedMatrixFactorization


def test_matrix_factorization_pipeline():
    toy_ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3, 3],
        "movieId": [10, 20, 30, 10, 20, 20, 30, 40],
        "rating": [5.0, 4.0, 2.0, 4.5, 4.0, 3.5, 1.0, 4.0],
    })

    model = BiasedMatrixFactorization(n_factors=4, n_epochs=10, random_state=42)
    model.fit(toy_ratings)

    preds = model.predict(user_ids=[1, 2], movie_ids=[10, 30])
    assert len(preds) == 2
    assert "predicted_score" in preds.columns
    assert "confidence" in preds.columns
    assert 0.5 <= preds["predicted_score"].iloc[0] <= 5.0

    cold_preds = model.predict(user_ids=[999, 1], movie_ids=[10, 9999])
    assert len(cold_preds) == 2
    assert cold_preds["confidence"].iloc[0] < 1.0

    print("All Matrix Factorization unit tests passed successfully!")


if __name__ == "__main__":
    test_matrix_factorization_pipeline()