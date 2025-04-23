# Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

data = load_diabetes()
X, y = data.data, data.target

y_binary = (y > np.median(y)).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(
    y_test,
    model.predict_proba(X_test)[:, 1],
)

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
sns.scatterplot(
    x=X_test[:, 2],
    y=X_test[:, 8],
    hue=y_test,
    palette={0: "blue", 1: "red"},
    marker="o",
)

plt.subplot(2, 1, 2)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")

plt.tight_layout()
plt.show()
