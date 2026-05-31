
import os
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please set it in the environment.")

api = HfApi(token=HF_TOKEN)

SPACE_REPO_ID = "RaginiPranay/predictive-maintenance-app"

# Create Hugging Face Space if it does not already exist
api.create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

files_to_upload = {
    "predictive_maintenance/deployment/app.py": "app.py",
    "predictive_maintenance/deployment/requirements.txt": "requirements.txt",
    "predictive_maintenance/deployment/Dockerfile": "Dockerfile"
}

for local_path, repo_path in files_to_upload.items():
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"Uploaded {local_path} to Hugging Face Space as {repo_path}")

print(f"Deployment files uploaded successfully to: https://huggingface.co/spaces/{SPACE_REPO_ID}")
