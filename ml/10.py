import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

plt.imshow(X_train[0])
plt.title(y_train[0])
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0


model = Sequential()
model.add(Input(shape=(28, 28)))
model.add(Flatten())
model.add(keras.layers.Dropout(rate=0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="adamw", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.show()

plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()
