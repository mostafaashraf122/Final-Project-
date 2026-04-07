from __future__ import annotations

from typing import Any

import pandas as pd

PROJECT_NAME = "Telecom Customer Churn Model Showcase"
PROJECT_TAGLINE = "A portfolio-style Streamlit experience for exploring and testing the saved churn model."
PROJECT_SUMMARY = (
    "This app focuses on the trained customer-churn model only. "
    "It loads the saved artifact from the project's artifacts folder, "
    "recreates the inference-ready feature row, and makes predictions "
    "through the same fitted pipeline used at training time."
)
PREDICTION_TASK = (
    "Predict whether a telecom customer profile is likely to churn "
    "(leave the service) or stay."
)
OLD_DEPLOYMENT_NOTE = (
    "The legacy `telecom_deployment.py` script is an exploratory-data-analysis dashboard. "
    "The new app is intentionally separate and centered on model inference."
)

TARGET_NAME = "churn"
TARGET_LABELS = {0: "No", 1: "Yes", "No": "No", "Yes": "Yes"}
TARGET_DESCRIPTIONS = {
    "No": "Customer is predicted to stay.",
    "Yes": "Customer is predicted to churn.",
}

MANUAL_INPUT_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]

ENGINEERED_FEATURES = ["cust-loyality", "family_member", "subscription_count"]

MODEL_INPUT_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
    "cust-loyality",
    "family_member",
    "subscription_count",
]

NUMERIC_INPUT_COLUMNS = ["tenure", "monthlycharges", "totalcharges", "subscription_count"]

SERVICE_COLUMNS = [
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
]

KNOWN_CATEGORY_LEVELS = {
    "gender": ["Female", "Male"],
    "seniorcitizen": ["old", "youth"],
    "partner": ["No", "Yes"],
    "dependents": ["No", "Yes"],
    "phoneservice": ["No", "Yes"],
    "multiplelines": ["No", "Yes", "unknown"],
    "internetservice": ["DSL", "Fiber optic", "No"],
    "onlinesecurity": ["No", "Yes"],
    "onlinebackup": ["No", "Yes"],
    "deviceprotection": ["No", "Yes"],
    "techsupport": ["No", "Yes"],
    "streamingtv": ["No", "Yes"],
    "streamingmovies": ["No", "Yes"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperlessbilling": ["No", "Yes"],
    "paymentmethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "cust-loyality": ["New", "Somewhat Loyal", "Loyal", "Very Loyal"],
    "family_member": [
        "Married",
        "Married With Dependents ",
        "Single",
        "Single With Dependents",
    ],
}

NUMERIC_BOUNDS = {
    "tenure": {"min": 1, "max": 72, "default": 35, "step": 1},
    "monthlycharges": {"min": 18.25, "max": 118.75, "default": 74.10, "step": 0.5},
    "totalcharges": {"min": 18.80, "max": 8684.80, "default": 1433.65, "step": 5.0},
    "subscription_count": {"min": 0, "max": 6, "default": 2, "step": 1},
}

FEATURE_GROUPS = {
    "Customer profile": ["gender", "seniorcitizen", "partner", "dependents", "tenure"],
    "Core services": ["phoneservice", "multiplelines", "internetservice", "contract"],
    "Add-on services": SERVICE_COLUMNS,
    "Billing": ["paperlessbilling", "paymentmethod", "monthlycharges", "totalcharges"],
}

FRIENDLY_LABELS = {
    "gender": "Gender",
    "seniorcitizen": "Senior-citizen band",
    "partner": "Has partner",
    "dependents": "Has dependents",
    "tenure": "Tenure (months)",
    "phoneservice": "Phone service",
    "multiplelines": "Multiple lines",
    "internetservice": "Internet service",
    "onlinesecurity": "Online security",
    "onlinebackup": "Online backup",
    "deviceprotection": "Device protection",
    "techsupport": "Tech support",
    "streamingtv": "Streaming TV",
    "streamingmovies": "Streaming movies",
    "contract": "Contract type",
    "paperlessbilling": "Paperless billing",
    "paymentmethod": "Payment method",
    "monthlycharges": "Monthly charges",
    "totalcharges": "Total charges",
    "cust-loyality": "Customer loyalty band",
    "family_member": "Family segment",
    "subscription_count": "Subscribed add-on count",
}

COLUMN_DESCRIPTIONS = {
    "gender": "Customer gender category used in the cleaned training dataset.",
    "seniorcitizen": "Binary customer age-band label from the project dataset (`old` or `youth`).",
    "partner": "Whether the customer has a partner/spouse.",
    "dependents": "Whether the customer has dependents.",
    "tenure": "Number of months the customer has stayed with the company.",
    "phoneservice": "Whether the customer has phone service.",
    "multiplelines": "Whether the customer has multiple phone lines. The cleaned project uses `unknown` when phone service is absent.",
    "internetservice": "Type of internet service in the cleaned dataset.",
    "onlinesecurity": "Whether the customer subscribes to online security.",
    "onlinebackup": "Whether the customer subscribes to online backup.",
    "deviceprotection": "Whether the customer subscribes to device protection.",
    "techsupport": "Whether the customer subscribes to tech support.",
    "streamingtv": "Whether the customer subscribes to streaming TV.",
    "streamingmovies": "Whether the customer subscribes to streaming movies.",
    "contract": "Contract duration category.",
    "paperlessbilling": "Whether billing is paperless.",
    "paymentmethod": "How the customer pays the bill.",
    "monthlycharges": "Current recurring monthly charge.",
    "totalcharges": "Accumulated total amount charged.",
    "cust-loyality": "Engineered tenure band created in the telecom data-preparation notebook.",
    "family_member": "Engineered family-status segment based on partner and dependents.",
    "subscription_count": "Engineered count of subscribed add-on internet services.",
}

PIPELINE_SUMMARY = [
    "Training data came from `cleaned_data.csv`, not the raw `data.csv` file.",
    "The modeling notebook used 18 categorical inputs and 4 numeric inputs.",
    "Numerical features were standardized with `StandardScaler`.",
    "Categorical features were encoded with `OneHotEncoder(handle_unknown='ignore')`.",
    "The saved artifact includes a fitted preprocessing pipeline and the trained estimator in one object.",
    "No separate scaler, encoder, or imputer artifact was saved outside the final joblib bundle.",
    "Manual app inputs recreate the three engineered modeling fields before inference: `cust-loyality`, `family_member`, and `subscription_count`.",
]

INFERENCE_ASSUMPTIONS = [
    "The saved joblib artifact is treated as the source of truth for inference.",
    "Predictions are only as reliable as the cleaned feature row supplied to the pipeline.",
    "If `internetservice` is `No`, the app forces the internet add-on services to `No` to match the cleaned training logic.",
    "If `phoneservice` is `No`, the app uses `multiplelines='unknown'`, which matches the cleaned dataset.",
    "Auto-generated class samples come from real rows in `cleaned_data.csv`, so they are realistic historical profiles rather than random values.",
]

LIMITATIONS = [
    "This is a binary churn classifier trained on historical telecom customer data; it does not explain causality.",
    "The cleaned dataset includes engineered fields and category labels from the project notebooks, including the `seniorcitizen` values `old` and `youth`.",
    "Feature-importance values come from the saved transformed-feature artifact and should be read as model influence, not business recommendations.",
    "The app does not retrain or recalibrate the model; it only loads the saved artifact for inference.",
    "Predictions outside the observed training ranges may be less trustworthy.",
]

USAGE_NOTES = [
    "Use Manual Input mode to build a customer profile yourself.",
    "Use Auto-Generated Sample mode to pull a realistic historical row intended for either class.",
    "Check the debug section to inspect the exact feature row passed into the model.",
]

NOTEBOOK_METRICS = {
    "train_rows": 475355,
    "test_rows": 118839,
    "positive_rate_train": 0.2252,
    "preprocessed_feature_count": 49,
    "selected_imbalance_strategy": "class_weight",
    "imbalance_sample_roc_auc_class_weight": 0.9160,
    "imbalance_sample_roc_auc_smoteenn": 0.9143,
    "baseline_model": "CatBoost",
    "baseline_validation_roc_auc": 0.9147,
    "selected_config": "tuned",
    "selected_validation_roc_auc": 0.9149,
    "mean_cv_roc_auc": 0.9149,
    "std_cv_roc_auc": 0.0010,
    "mean_cv_precision": 0.5591,
    "mean_cv_recall": 0.8784,
    "mean_cv_f1": 0.6833,
    "mean_cv_accuracy": 0.8166,
    "test_roc_auc": 0.9155,
    "test_precision": 0.5584,
    "test_recall": 0.8820,
    "test_f1": 0.6838,
    "test_accuracy": 0.8163,
}


def get_friendly_label(column_name: str) -> str:
    return FRIENDLY_LABELS.get(column_name, column_name.replace("_", " ").title())


def get_column_description(column_name: str) -> str:
    return COLUMN_DESCRIPTIONS.get(column_name, "")


def build_dataset_summary(reference_df: pd.DataFrame | None) -> dict[str, Any]:
    if reference_df is None or reference_df.empty:
        return {
            "row_count": None,
            "feature_count": len(MODEL_INPUT_COLUMNS),
            "churn_rate": None,
            "class_counts": {},
        }

    class_counts = reference_df[TARGET_NAME].value_counts().to_dict() if TARGET_NAME in reference_df.columns else {}
    churn_rate = None
    if TARGET_NAME in reference_df.columns:
        churn_rate = float((reference_df[TARGET_NAME] == "Yes").mean())

    return {
        "row_count": int(len(reference_df)),
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "churn_rate": churn_rate,
        "class_counts": class_counts,
    }


def build_rate_table(reference_df: pd.DataFrame | None, column_name: str) -> pd.DataFrame:
    if reference_df is None or reference_df.empty or column_name not in reference_df.columns or TARGET_NAME not in reference_df.columns:
        return pd.DataFrame(columns=[column_name, "customer_count", "churn_rate"])

    summary = (
        reference_df.assign(churn_flag=(reference_df[TARGET_NAME] == "Yes").astype(int))
        .groupby(column_name, dropna=False)["churn_flag"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "customer_count", "mean": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
        .reset_index(drop=True)
    )
    summary["churn_rate"] = summary["churn_rate"].round(4)
    return summary


def build_insight_cards(reference_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if reference_df is None or reference_df.empty:
        return []

    cards: list[dict[str, Any]] = []
    for column_name, title in [
        ("contract", "Highest churn contract"),
        ("internetservice", "Highest churn internet type"),
        ("paymentmethod", "Highest churn payment method"),
        ("cust-loyality", "Highest churn loyalty band"),
    ]:
        table = build_rate_table(reference_df, column_name)
        if table.empty:
            continue
        top_row = table.iloc[0]
        cards.append(
            {
                "title": title,
                "label": str(top_row[column_name]),
                "value": float(top_row["churn_rate"]),
                "count": int(top_row["customer_count"]),
            }
        )
    return cards


def get_metric_rows() -> list[dict[str, Any]]:
    return [
        {"metric": "Selected model", "value": NOTEBOOK_METRICS["baseline_model"]},
        {"metric": "Selected config", "value": NOTEBOOK_METRICS["selected_config"]},
        {"metric": "Validation ROC AUC", "value": NOTEBOOK_METRICS["selected_validation_roc_auc"]},
        {"metric": "Test ROC AUC", "value": NOTEBOOK_METRICS["test_roc_auc"]},
        {"metric": "Test precision", "value": NOTEBOOK_METRICS["test_precision"]},
        {"metric": "Test recall", "value": NOTEBOOK_METRICS["test_recall"]},
        {"metric": "Test F1", "value": NOTEBOOK_METRICS["test_f1"]},
        {"metric": "Test accuracy", "value": NOTEBOOK_METRICS["test_accuracy"]},
    ]
