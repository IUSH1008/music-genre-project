
import torch
import joblib
import pandas as pd
import os

from autoencoder import MusicAutoencoder

# Load prepared data
X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")

# Load trained autoencoder
model = MusicAutoencoder(input_dim=57, latent_dim=32)

model.load_state_dict(
    torch.load(
        "models/music_autoencoder.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Extract latent features
with torch.no_grad():
    Z_train = model.extract_latent(X_train_tensor).numpy()
    Z_test = model.extract_latent(X_test_tensor).numpy()

# Save latent features
os.makedirs("data/processed", exist_ok=True)

joblib.dump(Z_train, "data/processed/Z_train.pkl")
joblib.dump(Z_test, "data/processed/Z_test.pkl")

print("LATENT FEATURE EXTRACTION COMPLETED")
print("-----------------------------------")
print("Original training shape:", X_train.shape)
print("Latent training shape:", Z_train.shape)
print("Original testing shape:", X_test.shape)
print("Latent testing shape:", Z_test.shape)

print("\nCompression:")
print("57 original features -> 32 latent features")
print("Feature reduction:", round((1 - 32/57) * 100, 2), "%")

print("\nSaved latent features successfully!")
