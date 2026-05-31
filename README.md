
# Predictive Maintenance MLOps Pipeline

This project builds an end-to-end predictive maintenance pipeline for engine health monitoring.

The solution uses engine sensor readings such as engine RPM, lubrication oil pressure, fuel pressure, coolant pressure, lubrication oil temperature, and coolant temperature to predict engine condition.

## Project Workflow

1. Register raw data on Hugging Face Dataset Hub.
2. Prepare data and split it into train and test datasets.
3. Upload processed train and test datasets back to Hugging Face.
4. Train and compare machine learning models.
5. Track model parameters and evaluation metrics.
6. Register the best model on Hugging Face Model Hub.
7. Deploy the model using a Streamlit application.
8. Host the application on Hugging Face Spaces.
9. Automate the full workflow using GitHub Actions.

## Best Model

The best model selected during experimentation was AdaBoost. It achieved the highest F1-score for class 1 and also gave strong recall for identifying engines that may require maintenance.

## Repository Links

- Dataset Repository: https://huggingface.co/datasets/RaginiPranay/predictive-maintenance-dataset
- Model Repository: https://huggingface.co/RaginiPranay/predictive-maintenance-model
- Hugging Face Space: https://huggingface.co/spaces/RaginiPranay/predictive-maintenance-app
