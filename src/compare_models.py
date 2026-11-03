
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Results from our completed experiments
results = pd.DataFrame({
    "Model": [
        "Random Forest (Original)",
        "Autoencoder + RF (8D)",
        "Autoencoder + RF (16D)",
        "Autoencoder + RF (32D)"
    ],
    "Features": [57, 8, 16, 32],
    "Accuracy": [86.74, 71.97, 74.77, 78.38]
})

# Save comparison table
os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/model_comparison.csv", index=False)

# Create accuracy chart
plt.figure(figsize=(10, 6))
plt.bar(results["Model"], results["Accuracy"])
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Music Genre Classification - Model Comparison")
plt.ylim(0, 100)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig("outputs/model_comparison.png", dpi=300)
plt.close()

print("\nMODEL COMPARISON")
print("----------------")
print(results.to_string(index=False))

print("\nSaved:")
print("outputs/model_comparison.csv")
print("outputs/model_comparison.png")
