# for data manipulation
import pandas as pd
import os

# for train-test split
from sklearn.model_selection import train_test_split

# for Hugging Face authentication and uploads
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define dataset repo and input path
REPO_ID = "RaginiPranay/predictive-maintenance-dataset"
DATASET_PATH = f"hf://datasets/{REPO_ID}/engine_data.csv"

# Create local data folder if not already present
os.makedirs("predictive_maintenance/data", exist_ok=True)

# Step 1: Load dataset directly from Hugging Face dataset repo
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully from Hugging Face.")
print("Original shape:", df.shape)

# Step 2: Basic data cleaning
# Remove duplicate rows
duplicate_count = df.duplicated().sum()
print("Duplicate rows before removal:", duplicate_count)
df = df.drop_duplicates()

# Strip extra spaces from text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Check missing values
print("\nMissing values by column:")
print(df.isnull().sum())

# Since all columns are relevant to the prediction task, no columns are dropped

print("\nCleaned dataset shape:", df.shape)

# Step 3: Define features and target
target_col = "Engine Condition"

X = df.drop(columns=[target_col])
y = df[target_col]

# Step 4: Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(ytrain.value_counts(normalize=True).sort_index().mul(100).round(2))

print("\nTesting target distribution:")
print(ytest.value_counts(normalize=True).sort_index().mul(100).round(2))

print("\nTrain and test split completed.")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)

# Step 5: Save locally
Xtrain_path = "predictive_maintenance/data/Xtrain.csv"
Xtest_path = "predictive_maintenance/data/Xtest.csv"
ytrain_path = "predictive_maintenance/data/ytrain.csv"
ytest_path = "predictive_maintenance/data/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("\nFiles saved locally:")
print(Xtrain_path)
print(Xtest_path)
print(ytrain_path)
print(ytest_path)

# Step 6: Upload processed files back to Hugging Face dataset repo
files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded: {os.path.basename(file_path)}")

print(f"\nAll processed files uploaded successfully to: https://huggingface.co/datasets/{REPO_ID}")
