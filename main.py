import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("data/student_performance.csv")
df = df.dropna()

df["High_Performer"] = df["Exam_Score"].apply(lambda score: 1 if score >= 70 else 0)

X = df.drop(columns=["Exam_Score", "High_Performer"])
y = df["High_Performer"]

for column in X.select_dtypes(include=["object", "str"]).columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest accuracy:", accuracy)

predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype("int32")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_classes))

print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.savefig("model_accuracy.png")
plt.show()

model.save("models/student_performance_model.keras")

print("\nModel saved successfully.")