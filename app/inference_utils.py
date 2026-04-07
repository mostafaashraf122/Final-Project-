from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.model_metadata import (
    ENGINEERED_FEATURES,
    KNOWN_CATEGORY_LEVELS,
    MANUAL_INPUT_COLUMNS,
    MODEL_INPUT_COLUMNS,
    NUMERIC_BOUNDS,
    SERVICE_COLUMNS,
    TARGET_LABELS,
    TARGET_NAME,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_ARTIFACT = ARTIFACTS_DIR / "final_churn_model.joblib"
DEFAULT_FEATURE_IMPORTANCE_ARTIFACT = ARTIFACTS_DIR / "model_features.csv"
DEFAULT_REFERENCE_DATASET = PROJECT_ROOT / "cleaned_data.csv"


class ArtifactLoadError(RuntimeError):
    """Raised when the saved model artifact cannot be loaded safely."""


def resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_ARTIFACT) -> dict[str, Any]:
    artifact_path = _resolve_existing_path(model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Saved model artifact was not found: {artifact_path}")

    try:
        raw_bundle = joblib.load(artifact_path)
    except ModuleNotFoundError as exc:
        missing_package = getattr(exc, "name", "a required package")
        raise ArtifactLoadError(
            f"Unable to load the saved model because `{missing_package}` is not installed. "
            "Install the project requirements and try again."
        ) from exc
    except Exception as exc:
        if "unsupported pickle protocol" in str(exc):
            raise ArtifactLoadError(
                "The saved model artifact was created with a newer Python/joblib protocol. "
                "Use a modern Python environment, install the project requirements, and try again."
            ) from exc
        raise ArtifactLoadError(f"Unable to load the saved model artifact: {exc}") from exc

    pipeline = raw_bundle.get("pipeline") if isinstance(raw_bundle, dict) else raw_bundle
    if pipeline is None:
        raise ArtifactLoadError("The saved artifact did not contain a fitted `pipeline` object.")

    estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    preprocessed_feature_names: list[str] = []
    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        for name, _, columns in preprocessor.transformers_:
            if name == "num":
                numeric_features = list(columns)
            elif name == "cat":
                categorical_features = list(columns)
        if hasattr(preprocessor, "get_feature_names_out"):
            preprocessed_feature_names = list(preprocessor.get_feature_names_out())

    bundle = {
        "artifact_path": artifact_path,
        "raw_bundle": raw_bundle,
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "model_name": raw_bundle.get("model_name") if isinstance(raw_bundle, dict) else type(estimator).__name__,
        "selected_config": raw_bundle.get("selected_config") if isinstance(raw_bundle, dict) else None,
        "best_params": raw_bundle.get("best_params", {}) if isinstance(raw_bundle, dict) else {},
        "saved_preprocessed_features": raw_bundle.get("features", []) if isinstance(raw_bundle, dict) else [],
        "pipeline_type": type(pipeline).__name__,
        "estimator_type": type(estimator).__name__,
        "preprocessor_type": type(preprocessor).__name__ if preprocessor is not None else None,
        "supports_predict_proba": hasattr(pipeline, "predict_proba"),
        "supports_decision_function": hasattr(pipeline, "decision_function"),
        "raw_numeric_features": numeric_features,
        "raw_categorical_features": categorical_features,
        "raw_feature_columns": MODEL_INPUT_COLUMNS.copy(),
        "preprocessed_feature_names": preprocessed_feature_names,
    }
    return bundle


def load_feature_importance(feature_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_ARTIFACT) -> pd.DataFrame:
    artifact_path = _resolve_existing_path(feature_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Feature-importance artifact was not found: {artifact_path}")
    return pd.read_csv(artifact_path)


def load_reference_dataset(dataset_path: str | Path = DEFAULT_REFERENCE_DATASET) -> pd.DataFrame:
    path = _resolve_existing_path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference dataset was not found: {path}")
    return pd.read_csv(path, index_col=0)


def get_category_options(reference_df: pd.DataFrame | None, column_name: str) -> list[str]:
    if reference_df is not None and column_name in reference_df.columns:
        values = reference_df[column_name].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)
    return KNOWN_CATEGORY_LEVELS.get(column_name, [])


def build_manual_defaults(reference_df: pd.DataFrame | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    if reference_df is None or reference_df.empty:
        for column_name in MANUAL_INPUT_COLUMNS:
            if column_name in KNOWN_CATEGORY_LEVELS:
                defaults[column_name] = KNOWN_CATEGORY_LEVELS[column_name][0]
            else:
                defaults[column_name] = NUMERIC_BOUNDS[column_name]["default"]
        return defaults

    for column_name in MANUAL_INPUT_COLUMNS:
        if column_name in KNOWN_CATEGORY_LEVELS:
            mode = reference_df[column_name].mode(dropna=True)
            defaults[column_name] = str(mode.iloc[0]) if not mode.empty else KNOWN_CATEGORY_LEVELS[column_name][0]
        else:
            median = reference_df[column_name].median()
            defaults[column_name] = float(median)

    defaults["tenure"] = int(round(defaults["tenure"]))
    return defaults


def derive_customer_loyalty(tenure: int) -> str:
    if tenure < 12:
        return "New"
    if tenure < 24:
        return "Somewhat Loyal"
    if tenure <= 48:
        return "Loyal"
    return "Very Loyal"


def derive_family_member(partner: str, dependents: str) -> str:
    if partner == "No" and dependents == "No":
        return "Single"
    if partner == "Yes" and dependents == "Yes":
        return "Married With Dependents "
    if partner == "No" and dependents == "Yes":
        return "Single With Dependents"
    return "Married"


def derive_subscription_count(row: dict[str, Any]) -> int:
    if row.get("onlinesecurity") == "unknown":
        return 0
    return sum(1 for column_name in SERVICE_COLUMNS if row.get(column_name) == "Yes")


def normalize_manual_input(base_inputs: dict[str, Any]) -> dict[str, Any]:
    row = dict(base_inputs)
    row["tenure"] = int(row["tenure"])
    row["monthlycharges"] = round(float(row["monthlycharges"]), 2)
    row["totalcharges"] = round(float(row["totalcharges"]), 2)

    if row.get("phoneservice") == "No":
        row["multiplelines"] = "unknown"

    if row.get("internetservice") == "No":
        for service_name in SERVICE_COLUMNS:
            row[service_name] = "No"

    row["cust-loyality"] = derive_customer_loyalty(row["tenure"])
    row["family_member"] = derive_family_member(str(row["partner"]), str(row["dependents"]))
    row["subscription_count"] = derive_subscription_count(row)

    missing_columns = [column_name for column_name in MODEL_INPUT_COLUMNS if column_name not in row]
    if missing_columns:
        raise ValueError(f"Missing required inference inputs: {missing_columns}")

    return {column_name: row[column_name] for column_name in MODEL_INPUT_COLUMNS}


def build_inference_frame(base_inputs: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([normalize_manual_input(base_inputs)])


def sample_reference_row(
    reference_df: pd.DataFrame,
    intended_class: str,
    random_state: int | None = None,
) -> dict[str, Any]:
    if TARGET_NAME not in reference_df.columns:
        raise ValueError(f"Reference dataset is missing `{TARGET_NAME}`.")

    subset = reference_df.loc[reference_df[TARGET_NAME] == intended_class].copy()
    if subset.empty:
        raise ValueError(f"No rows were found for intended class `{intended_class}`.")

    sampled_row = subset.sample(n=1, random_state=random_state).iloc[0]
    model_row = sampled_row.drop(labels=[col for col in ["id", TARGET_NAME] if col in sampled_row.index])
    model_row = model_row.reindex(MODEL_INPUT_COLUMNS)

    return {
        "intended_class": intended_class,
        "source_row": sampled_row.to_dict(),
        "model_row": model_row.to_dict(),
        "model_frame": pd.DataFrame([model_row.to_dict()]),
    }


def coerce_prediction_label(prediction: Any) -> str:
    if prediction in TARGET_LABELS:
        return TARGET_LABELS[prediction]
    return str(prediction)


def predict_record(bundle: dict[str, Any], input_frame: pd.DataFrame) -> dict[str, Any]:
    pipeline = bundle["pipeline"]
    schema_warnings = compare_frame_to_schema(input_frame)

    prediction_value = pipeline.predict(input_frame)[0]
    predicted_label = coerce_prediction_label(prediction_value)

    classes = list(getattr(pipeline, "classes_", getattr(bundle["estimator"], "classes_", [])))
    probability_map: dict[str, float] = {}
    if bundle.get("supports_predict_proba"):
        probabilities = pipeline.predict_proba(input_frame)[0]
        probability_map = {
            coerce_prediction_label(class_name): float(probability)
            for class_name, probability in zip(classes, probabilities)
        }

    decision_score = None
    if bundle.get("supports_decision_function"):
        score = pipeline.decision_function(input_frame)
        if hasattr(score, "__len__") and not isinstance(score, (str, bytes)):
            decision_score = float(score[0])
        else:
            decision_score = float(score)

    predicted_probability = probability_map.get(predicted_label)
    yes_probability = probability_map.get("Yes")

    summary_sentence = (
        f"The saved model predicts `{predicted_label}` for this customer profile."
        if predicted_probability is None
        else f"The saved model predicts `{predicted_label}` with {predicted_probability:.1%} confidence."
    )
    if yes_probability is not None:
        summary_sentence += f" Estimated churn probability: {yes_probability:.1%}."

    return {
        "predicted_value": prediction_value,
        "predicted_label": predicted_label,
        "probabilities": probability_map,
        "decision_score": decision_score,
        "predicted_probability": predicted_probability,
        "yes_probability": yes_probability,
        "summary_sentence": summary_sentence,
        "input_frame": input_frame.copy(),
        "schema_warnings": schema_warnings,
    }


def compare_frame_to_schema(input_frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    missing_columns = [column_name for column_name in MODEL_INPUT_COLUMNS if column_name not in input_frame.columns]
    extra_columns = [column_name for column_name in input_frame.columns if column_name not in MODEL_INPUT_COLUMNS]

    if missing_columns:
        warnings.append(f"Missing expected columns: {missing_columns}")
    if extra_columns:
        warnings.append(f"Unexpected extra columns: {extra_columns}")

    if not missing_columns and not extra_columns:
        ordered_columns = list(input_frame.columns)
        if ordered_columns != MODEL_INPUT_COLUMNS:
            warnings.append("Input columns are valid but not in the original training order.")

    return warnings


def build_debug_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": str(bundle["artifact_path"]),
        "model_name": bundle.get("model_name"),
        "selected_config": bundle.get("selected_config"),
        "best_params": bundle.get("best_params") or {},
        "pipeline_type": bundle.get("pipeline_type"),
        "preprocessor_type": bundle.get("preprocessor_type"),
        "estimator_type": bundle.get("estimator_type"),
        "supports_predict_proba": bundle.get("supports_predict_proba"),
        "supports_decision_function": bundle.get("supports_decision_function"),
        "raw_numeric_features": bundle.get("raw_numeric_features", []),
        "raw_categorical_features": bundle.get("raw_categorical_features", []),
        "raw_feature_columns": bundle.get("raw_feature_columns", []),
        "preprocessed_feature_count": len(bundle.get("preprocessed_feature_names") or bundle.get("saved_preprocessed_features") or []),
    }


def build_inference_ready_preview(base_inputs: dict[str, Any]) -> pd.DataFrame:
    frame = build_inference_frame(base_inputs)
    display_columns = MANUAL_INPUT_COLUMNS + ENGINEERED_FEATURES
    return frame[display_columns]
