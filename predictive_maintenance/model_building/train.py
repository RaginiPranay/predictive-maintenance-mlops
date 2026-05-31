
# For data handling
import os
import pandas as pd
import joblib

# For Hugging Face download and upload
from huggingface_hub import hf_hub_download, HfApi

# For model building
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# For evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Hugging Face details
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Please make sure it is set in the environment.")

DATASET_REPO_ID = "RaginiPranay/predictive-maintenance-dataset"
MODEL_REPO_ID = "RaginiPranay/predictive-maintenance-model"

api = HfApi(token=HF_TOKEN)

# Create local folders
os.makedirs("predictive_maintenance/model_artifacts", exist_ok=True)
os.makedirs("predictive_maintenance/experiment_logs", exist_ok=True)


# Helper function to download files from Hugging Face dataset repo
# It first checks the root path. If not found, it checks inside the data folder.
def download_dataset_file(filename):
    try:
        return hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception:
        return hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=f"data/{filename}",
            repo_type="dataset",
            token=HF_TOKEN
        )


# Step 1: Load train and test data from Hugging Face
Xtrain_path = download_dataset_file("Xtrain.csv")
Xtest_path = download_dataset_file("Xtest.csv")
ytrain_path = download_dataset_file("ytrain.csv")
ytest_path = download_dataset_file("ytest.csv")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print("Train and test data loaded successfully from Hugging Face.")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)


# Step 2: Define models and tuned parameter settings
experiments = [
    {
        "model_name": "Decision Tree",
        "model": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=5
        ),
        "parameters": {
            "max_depth": 5,
            "min_samples_leaf": 5,
            "class_weight": "balanced"
        }
    },
    {
        "model_name": "Decision Tree",
        "model": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=7,
            min_samples_leaf=10
        ),
        "parameters": {
            "max_depth": 7,
            "min_samples_leaf": 10,
            "class_weight": "balanced"
        }
    },
    {
        "model_name": "Random Forest",
        "model": RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=1
        ),
        "parameters": {
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced"
        }
    },
    {
        "model_name": "Random Forest",
        "model": RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=1
        ),
        "parameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 5,
            "class_weight": "balanced"
        }
    },
    {
        "model_name": "AdaBoost",
        "model": AdaBoostClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.5
        ),
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.5
        }
    },
    {
        "model_name": "Gradient Boosting",
        "model": GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        ),
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    }
]


# Step 3: Train, tune, log, and evaluate models
experiment_results = []
best_model = None
best_f1 = -1

for experiment in experiments:
    model_name = experiment["model_name"]
    model = experiment["model"]
    parameters = experiment["parameters"]

    print("\nTraining model:", model_name)
    print("Parameters:", parameters)

    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred, pos_label=1)
    recall = recall_score(ytest, y_pred, pos_label=1)
    f1 = f1_score(ytest, y_pred, pos_label=1)

    cm = confusion_matrix(ytest, y_pred)
    tn, fp, fn, tp = cm.ravel()

    result = {
        "Model": model_name,
        "Parameters": parameters,
        "Accuracy": round(accuracy, 4),
        "Precision_Class_1": round(precision, 4),
        "Recall_Class_1": round(recall, 4),
        "F1_Class_1": round(f1, 4),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    experiment_results.append(result)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Confusion Matrix:")
    print(cm)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_result = result


# Step 4: Save experiment tracking log
experiment_log_df = pd.DataFrame(experiment_results)

experiment_log_path = "predictive_maintenance/experiment_logs/experiment_tracking_log.csv"
experiment_log_df.to_csv(experiment_log_path, index=False)

print("\nExperiment tracking log saved at:", experiment_log_path)
print("\nExperiment Results:")
print(experiment_log_df)


# Step 5: Save best model locally
best_model_path = "predictive_maintenance/model_artifacts/best_model.pkl"
joblib.dump(best_model, best_model_path)

print("\nBest Model Selected:")
print(best_result)

print("\nBest model saved locally at:", best_model_path)


# Step 6: Register best model and logs in Hugging Face Model Hub
api.create_repo(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=best_model_path,
    path_in_repo="best_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj=experiment_log_path,
    path_in_repo="experiment_tracking_log.csv",
    repo_id=MODEL_REPO_ID,
    repo_type="model"
)

print(f"\nBest model and experiment log uploaded successfully to: https://huggingface.co/{MODEL_REPO_ID}")
