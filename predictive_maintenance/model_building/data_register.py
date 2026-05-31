from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "RaginiPranay/predictive-maintenance-dataset"
repo_type = "dataset"

# Initialize API client using HF token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created successfully.")

# Step 2: Upload local data folder to Hugging Face dataset repo
api.upload_folder(
    folder_path="predictive_maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"Data uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
