
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import time

from autoencoder import MusicAutoencoder

start_time = time.time()

# Use CPU
device = torch.device("cpu")

# Load training and testing data
X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")

# Convert data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Create batches
dataset = TensorDataset(X_train)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

# Create model
model = MusicAutoencoder(input_dim=57, latent_dim=32).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training settings
epochs = 50

print("Starting Autoencoder Training...")
print("--------------------------------")

# Training loop
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for (batch,) in loader:

        batch = batch.to(device)

        # Forward pass
        reconstructed = model(batch)

        # Calculate reconstruction loss
        loss = criterion(reconstructed, batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.6f}"
        )

# Evaluate reconstruction loss on test data
model.eval()

with torch.no_grad():
    test_reconstructed = model(X_test)
    test_loss = criterion(test_reconstructed, X_test)

print("\nTRAINING COMPLETED")
print("------------------")
print(f"Test Reconstruction Loss: {test_loss.item():.6f}")

# Save trained model
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/music_autoencoder.pth"
)

print("Model saved: models/music_autoencoder.pth")
print(f"Total time: {time.time() - start_time:.2f} seconds")
