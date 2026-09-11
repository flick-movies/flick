import numpy as np


def mae(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def accuracy_within(actual, predicted, tolerance=0.5):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(
        np.mean(np.abs(actual - predicted) <= tolerance)
    )