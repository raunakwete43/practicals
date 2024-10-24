from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


# Naive Bayes Classifier
class NaiveBayes:
    def __init__(self) -> None:
        self.class_priors: Dict[str, float] = {}
        self.feature_likelihoods: Dict[str, Dict[str, Dict[str, float]]] = {}

    def fit(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series]
    ) -> None:
        self.classes: np.ndarray = np.unique(y)
        for c in self.classes:
            # Calculate class prior probability P(C)
            self.class_priors[str(c)] = np.mean(y == c)

            # Calculate likelihoods P(X|C) for each feature given the class
            self.feature_likelihoods[str(c)] = {}
            for feature in X.columns:
                self.feature_likelihoods[str(c)][feature] = {}
                for value in np.unique(X[feature]):
                    # Calculate the likelihood: P(feature=value|class=c)
                    feature_given_class = X[(X[feature] == value) & (y == c)]
                    likelihood = len(feature_given_class) / np.sum(y == c)
                    self.feature_likelihoods[str(c)][feature][str(value)] = likelihood

    def predict(self, X: pd.DataFrame) -> Tuple[List[str], List[Dict[str, float]]]:
        predictions: List[str] = []
        probabilities: List[Dict[str, float]] = []

        for _, row in X.iterrows():
            posteriors: Dict[str, float] = {}
            for c in self.classes:
                class_str = str(c)
                # Start with class prior
                posterior = self.class_priors[class_str]
                for feature in X.columns:
                    value = str(
                        row[feature]
                    )  # Ensure value is a string for dict access
                    # Multiply by the likelihood P(X|C)
                    posterior *= self.feature_likelihoods[class_str][feature].get(
                        value, 1e-6
                    )
                posteriors[class_str] = posterior
            # Choose class with highest posterior
            predicted_class = max(posteriors, key=lambda k: posteriors[k])
            predictions.append(predicted_class)
            # Normalize to get probabilities
            total = sum(posteriors.values())
            prob = {k: float(v / total) for k, v in posteriors.items()}
            probabilities.append(prob)

        return predictions, probabilities


# Create a dummy dataset
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

# Create a DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df[["Outlook", "Temperature", "Humidity", "Windy"]]
y = df["Play Golf"]

# Create an instance of NaiveBayes and fit the model
nb = NaiveBayes()
nb.fit(X, y)

# Test the classifier with new data
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
print(nb.feature_likelihoods)
