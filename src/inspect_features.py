import pandas as pd
import time

start = time.time()

file_path = "data/raw/Data/features_3_sec.csv"

df = pd.read_csv(file_path)

print("\nDATASET INFORMATION")
print("-------------------")

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nGenre distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nInspection completed!")
print("Execution time:", round(time.time() - start, 2), "seconds")