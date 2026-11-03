
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("data/raw/Data/features_3_sec.csv")

# Separate input features and target labels
X = df.drop(columns=["filename", "length", "label"])
y = df["label"]

# Convert genre names into numerical labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# Save processed data
joblib.dump(X_train_scaled, "data/processed/X_train.pkl")
joblib.dump(X_test_scaled, "data/processed/X_test.pkl")
joblib.dump(y_train, "data/processed/y_train.pkl")
joblib.dump(y_test, "data/processed/y_test.pkl")
joblib.dump(scaler, "data/processed/scaler.pkl")
joblib.dump(encoder, "data/processed/label_encoder.pkl")

print("DATA PREPARATION COMPLETED")
print("--------------------------------")
print("Training samples:", X_train_scaled.shape)
print("Testing samples:", X_test_scaled.shape)
print("Number of features:", X_train_scaled.shape[1])
print("Genres:", list(encoder.classes_))
print("Files saved in data/processed/")
