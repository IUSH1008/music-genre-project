
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load latent features
Z_train = joblib.load("data/processed/Z_train.pkl")
Z_test = joblib.load("data/processed/Z_test.pkl")

# Load labels
y_train = joblib.load("data/processed/y_train.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Load genre encoder
encoder = joblib.load("data/processed/label_encoder.pkl")

print("Training classifier on latent features...")
print("-----------------------------------------")

# Create classifier
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(Z_train, y_train)

# Predict
y_pred = model.predict(Z_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nLATENT FEATURE MODEL RESULTS")
print("----------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# Save classifier
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/latent_random_forest.pkl"
)

print("\nModel saved to models/latent_random_forest.pkl")
