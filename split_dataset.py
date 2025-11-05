#ONLY WANT TO SPLIT THE DATASET INTO TRAIN AND VAL SETS BECAUSE I AM CREATING MY OWN TESTING DATASET
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("code_snippets.csv")

# First: split into train (80%) and temp (20%)
train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["language"]
)

# Second: split temp into validation (13%) and test (7%)
# 13% and 7% relative to total means validation = 0.13/0.20 = 0.65 of temp
val_df, test_df = train_test_split(
    temp_df, test_size=0.35, random_state=42, stratify=temp_df["language"]
)

# Save the splits
train_df.to_csv("train_snippets.csv", index=False)
val_df.to_csv("val_snippets.csv", index=False)
test_df.to_csv("test_snippets.csv", index=False)

print("Training set:", len(train_df))
print("Validation set:", len(val_df))
print("Testing set:", len(test_df))

