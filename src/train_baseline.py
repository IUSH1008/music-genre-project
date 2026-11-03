
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load prepared data
X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
y_test = joblib.load("data/processed/y_test.pkl")
encoder = joblib.load("data/processed/label_encoder.pkl")

print("Training Random Forest model...")

# Create and train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict genres
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nBASELINE MODEL RESULTS")
print("----------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_baseline.pkl")

print("\nModel saved to models/random_forest_baseline.pkl")
