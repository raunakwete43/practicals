from typing import List, Union

import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self) -> None:
        self.class_priors: dict[str, float] = {}
        self.feature_likelihoods: dict[str, dict[str, dict[str, float]]] = {}

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series]):
        self.classes: np.ndarray = np.unique(y)

        for c in self.classes:
            self.class_priors[str(c)] = np.mean(y == c)

            self.feature_likelihoods[str(c)] = {}
            for feature in X.columns:
                self.feature_likelihoods[str(c)][feature] = {}

                for value in np.unique(X[feature]):
                    feature_given_class = X[(X[feature] == value) & (y == c)]
                    likelihood = len(feature_given_class) / np.sum(y == c)
                    self.feature_likelihoods[str(c)][feature][str(value)] = likelihood

    def predict(self, X: Union[pd.DataFrame, pd.Series]):
        predictions: List[str] = []
        probabilities: List[dict[str, float]] = []

        for _, row in X.iterrows():
            posteriors: dict[str, float] = {}
            for c in self.classes:
                class_str = str(c)

                posterior = self.class_priors[class_str]

                for feature in X.columns:
                    value = str(row[feature])
                    posterior *= self.feature_likelihoods[class_str][feature].get(
                        value, 1e-6
                    )

                posteriors[class_str] = posterior

            predicted_class = max(posteriors, key=lambda x: posteriors[x])
            predictions.append(predicted_class)

            total = sum(posteriors.values())
            prob = {k: float(v / total) for k, v in posteriors.items()}
            probabilities.append(prob)

        return predictions, probabilities


data = {
    "Outlook": [
        "Rainy",
        "Rainy",
        "Overcast",
        "Sunny",
        "Sunny",
        "Sunny",
        "Overcast",
        "Rainy",
        "Rainy",
        "Sunny",
        "Rainy",
        "Overcast",
        "Overcast",
        "Sunny",
    ],
    "Temperature": [
        "Hot",
        "Hot",
        "Hot",
        "Mild",
        "Cool",
        "Cool",
        "Cool",
        "Mild",
        "Cool",
        "Mild",
        "Mild",
        "Mild",
        "Hot",
        "Mild",
    ],
    "Humidity": [
        "High",
        "High",
        "High",
        "High",
        "Normal",
        "Normal",
        "Normal",
        "High",
        "Normal",
        "Normal",
        "Normal",
        "High",
        "Normal",
        "High",
    ],
    "Windy": [
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
    ],
    "Play Golf": [
        "No",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "No",
    ],
}

df = pd.DataFrame(data)

X = df[["Outlook", "Temperature", "Humidity", "Windy"]]
y = df["Play Golf"]

nb = NaiveBayes()
nb.fit(X, y)

new_data = pd.DataFrame(
    {
        "Outlook": ["Sunny"],
        "Temperature": ["Hot"],
        "Humidity": ["Normal"],
        "Windy": [False],
    }
)

predictions, probabilities = nb.predict(new_data)
print("Probabilities: ", probabilities)
print("Predictions: ", predictions)
