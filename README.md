# Telecom Customer Churn Project

This repository contains a telecom customer churn project with three clearly separated paths:

1. `telecom.ipynb`
   The original data-understanding, cleaning, feature-engineering, and EDA notebook.
2. `telecom_deployment.py`
   The original Streamlit dashboard for exploratory analysis over `cleaned_data.csv`.
3. `app/model_streamlit_app.py`
   The new model-focused portfolio app for inference, testing, and model explanation.

The new app is intentionally separate from the old EDA deployment path. It loads the saved trained model from `artifacts/final_churn_model.joblib`, recreates the inference-ready feature row, supports manual testing, and can generate realistic sample profiles for both churn classes.

## Prediction Task

The model predicts whether a telecom customer is likely to churn:

- `No`: customer is predicted to stay
- `Yes`: customer is predicted to churn

The training notebook (`ML .ipynb`) shows that the final selected model is a tuned CatBoost pipeline evaluated with:

- Validation ROC AUC: `0.9149`
- Test ROC AUC: `0.9155`
- Test precision: `0.5584`
- Test recall: `0.8820`
- Test F1: `0.6838`
- Test accuracy: `0.8163`

## Current Project Layout

```text
.
├── app/
│   ├── __init__.py
│   ├── inference_utils.py
│   ├── model_metadata.py
│   └── model_streamlit_app.py
├── artifacts/
│   ├── final_churn_model.joblib
│   └── model_features.csv
├── notebooks/
│   └── model_deployment_testing.ipynb
├── ML .ipynb
├── telecom.ipynb
├── telecom_deployment.py
├── cleaned_data.csv
├── data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

Historical notebooks remain in the project root so their existing relative paths do not break.

## Old vs New Deployment Paths

### Old path

- `telecom.ipynb` prepares and explores the data.
- `telecom_deployment.py` is a Streamlit EDA dashboard.
- This path is useful for dataset understanding and visual analysis.

### New path

- `app/model_streamlit_app.py` is a dedicated model showcase and testing experience.
- `app/inference_utils.py` handles model loading, schema preparation, feature derivation, prediction, and sample generation.
- `app/model_metadata.py` centralizes grounded model notes, metrics, labels, and dataset summaries.
- `notebooks/model_deployment_testing.ipynb` mirrors the inference workflow in notebook form.

## How Inference Works in This Project

The modeling notebook saved a joblib bundle that contains:

- `model_name`
- `selected_config`
- `best_params`
- `pipeline`
- transformed feature names

The saved `pipeline` already includes preprocessing and the final trained estimator. That means:

- numeric features are scaled inside the pipeline
- categorical features are one-hot encoded inside the pipeline
- no separate scaler or encoder file is required for inference

The training notebook modeled these cleaned features:

- Base inputs: `gender`, `seniorcitizen`, `partner`, `dependents`, `tenure`, `phoneservice`, `multiplelines`, `internetservice`, `onlinesecurity`, `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`, `contract`, `paperlessbilling`, `paymentmethod`, `monthlycharges`, `totalcharges`
- Engineered fields: `cust-loyality`, `family_member`, `subscription_count`

The new app computes the three engineered fields automatically for manual input mode so the final inference row matches the modeling schema.

## Install Dependencies

Create and activate a virtual environment, then install the project requirements.

Recommended: use Python `3.11` or `3.12` for the smoothest CatBoost installation experience. The saved model artifact was produced with `scikit-learn 1.6.1`, so the requirements file keeps that version pinned for inference compatibility.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Notebooks

You can open the notebooks in VS Code or Jupyter.

- `telecom.ipynb`: data cleaning, feature engineering, and EDA
- `ML .ipynb`: modeling, evaluation, and artifact saving
- `notebooks/model_deployment_testing.ipynb`: new inference/deployment walkthrough

If you want to start Jupyter manually:

```powershell
jupyter lab
```

## Run the New Streamlit App

From the project root:

```powershell
streamlit run app/model_streamlit_app.py
```

The app loads the trained model from:

```text
artifacts/final_churn_model.joblib
```

It also uses:

```text
artifacts/model_features.csv
cleaned_data.csv
```

## Example Manual Testing Flow

1. Open the Streamlit app.
2. Go to the `Manual Input` tab.
3. Enter or adjust the customer profile fields.
4. Optionally let the app estimate `totalcharges` from tenure and monthly charges.
5. Review the exact inference-ready row, including the automatically derived fields.
6. Click `Predict from manual input`.
7. Review the predicted class, class probabilities, and the full row sent to the pipeline.

## Example Auto-Generated Sample Flow

1. Open the `Auto-Generated Sample` tab.
2. Choose the intended class: `No` or `Yes`.
3. Click `Generate sample profile`.
4. Review the generated historical example before prediction.
5. Click `Predict generated sample`.
6. Compare the intended class against the predicted class and inspect the probabilities.

## Notes and Caveats

- The new app is model-focused and does not replace the original EDA dashboard.
- Predictions rely on the saved joblib artifact and the cleaned modeling schema.
- Feature-importance values are model influence indicators, not causal explanations.
- The cleaned dataset includes project-specific engineered categories such as `cust-loyality` and `family_member`.

## GitHub Push Commands

This folder was **not** initialized as a git repository when inspected. Use the commands below from the project root after reviewing your files:

```powershell
git init
git branch -M main
git add .
git commit -m "Add model-focused Streamlit showcase and inference notebook"
git remote add origin <MY_GITHUB_REPO_URL>
git push -u origin main
```

If you already create the GitHub repo first and it contains no commits, these commands will work as-is.
