from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.inference_utils import (  # noqa: E402
    DEFAULT_FEATURE_IMPORTANCE_ARTIFACT,
    DEFAULT_MODEL_ARTIFACT,
    DEFAULT_REFERENCE_DATASET,
    ArtifactLoadError,
    build_debug_summary,
    build_inference_ready_preview,
    build_manual_defaults,
    build_inference_frame,
    get_category_options,
    load_feature_importance,
    load_model_bundle,
    load_reference_dataset,
    predict_record,
    sample_reference_row,
)
from app.model_metadata import (  # noqa: E402
    FEATURE_GROUPS,
    FRIENDLY_LABELS,
    INFERENCE_ASSUMPTIONS,
    KNOWN_CATEGORY_LEVELS,
    LIMITATIONS,
    MANUAL_INPUT_COLUMNS,
    MODEL_INPUT_COLUMNS,
    NOTEBOOK_METRICS,
    NUMERIC_BOUNDS,
    OLD_DEPLOYMENT_NOTE,
    PIPELINE_SUMMARY,
    PREDICTION_TASK,
    PROJECT_NAME,
    PROJECT_SUMMARY,
    PROJECT_TAGLINE,
    TARGET_DESCRIPTIONS,
    TARGET_LABELS,
    USAGE_NOTES,
    build_dataset_summary,
    build_insight_cards,
    build_rate_table,
    get_column_description,
)

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(36, 99, 235, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 26%),
                linear-gradient(180deg, #f7fafc 0%, #eef4f7 100%);
        }
        .hero-card, .section-card, .metric-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            padding: 1.1rem 1.25rem;
        }
        .hero-card {
            padding: 1.4rem 1.5rem;
            margin-bottom: 0.8rem;
        }
        .eyebrow {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
            line-height: 1.15;
        }
        .hero-subtitle {
            color: #334155;
            font-size: 1.02rem;
            margin-bottom: 0.2rem;
        }
        .metric-label {
            color: #475569;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .metric-note {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin: 1.2rem 0 0.15rem 0;
        }
        .section-copy {
            color: #475569;
            margin-bottom: 0.7rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.8rem 0.95rem;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_model_bundle()


@st.cache_data(show_spinner=False)
def get_reference_data() -> pd.DataFrame:
    return load_reference_dataset()


@st.cache_data(show_spinner=False)
def get_feature_importance_data() -> pd.DataFrame:
    return load_feature_importance()


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle: dict | None, dataset_summary: dict, bundle_error: Exception | None) -> None:
    st.sidebar.title("Model Demo")
    st.sidebar.caption("New model-focused deployment path")
    st.sidebar.info(OLD_DEPLOYMENT_NOTE)

    st.sidebar.markdown("**Artifacts used**")
    st.sidebar.code(str(DEFAULT_MODEL_ARTIFACT), language="text")
    st.sidebar.code(str(DEFAULT_REFERENCE_DATASET), language="text")

    st.sidebar.markdown("**Notebook-grounded metrics**")
    st.sidebar.write(f"Selected model: `{NOTEBOOK_METRICS['baseline_model']}`")
    st.sidebar.write(f"Validation ROC AUC: `{NOTEBOOK_METRICS['selected_validation_roc_auc']:.4f}`")
    st.sidebar.write(f"Test ROC AUC: `{NOTEBOOK_METRICS['test_roc_auc']:.4f}`")
    st.sidebar.write(f"Test recall: `{NOTEBOOK_METRICS['test_recall']:.4f}`")

    if dataset_summary.get("row_count"):
        st.sidebar.markdown("**Reference dataset**")
        st.sidebar.write(f"Rows: `{dataset_summary['row_count']:,}`")
        st.sidebar.write(f"Observed churn rate: `{format_pct(dataset_summary['churn_rate'])}`")

    if bundle is not None:
        st.sidebar.success("Saved model artifact loaded successfully.")
    elif bundle_error is not None:
        st.sidebar.warning(str(bundle_error))


def render_hero(bundle: dict | None, dataset_summary: dict) -> None:
    class_labels = ", ".join(sorted({TARGET_LABELS[0], TARGET_LABELS[1]}))
    model_name = bundle.get("model_name") if bundle else NOTEBOOK_METRICS["baseline_model"]
    preprocessed_feature_count = (
        len(bundle.get("preprocessed_feature_names") or bundle.get("saved_preprocessed_features") or [])
        if bundle
        else NOTEBOOK_METRICS["preprocessed_feature_count"]
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Portfolio ML Demo</div>
            <div class="hero-title">{PROJECT_NAME}</div>
            <div class="hero-subtitle">{PROJECT_TAGLINE}</div>
            <div class="hero-subtitle">{PREDICTION_TASK}</div>
            <div class="hero-subtitle">{PROJECT_SUMMARY}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Selected model", str(model_name), "Saved artifact or notebook metadata")
    with metric_cols[1]:
        render_metric_card("Target labels", class_labels, "Binary churn outcome")
    with metric_cols[2]:
        render_metric_card("Test ROC AUC", f"{NOTEBOOK_METRICS['test_roc_auc']:.4f}", "From `ML .ipynb`")
    with metric_cols[3]:
        row_count = f"{dataset_summary['row_count']:,}" if dataset_summary.get("row_count") else "N/A"
        render_metric_card("Reference rows", row_count, f"{preprocessed_feature_count} transformed features")


def render_about_model(bundle: dict | None, dataset_summary: dict) -> None:
    st.markdown('<div class="section-title">About the Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Grounded project details extracted from the training notebook, artifacts, and cleaned dataset.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.15, 0.85])
    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        about_rows = [
            {"Field": "Prediction task", "Value": PREDICTION_TASK},
            {"Field": "Target variable", "Value": "churn"},
            {"Field": "Class labels", "Value": "No (stay), Yes (churn)"},
            {"Field": "Model family", "Value": str(bundle.get("model_name") if bundle else NOTEBOOK_METRICS["baseline_model"])},
            {"Field": "Selected configuration", "Value": str(bundle.get("selected_config") if bundle else NOTEBOOK_METRICS["selected_config"])},
            {"Field": "Training rows", "Value": f"{NOTEBOOK_METRICS['train_rows']:,}"},
            {"Field": "Test rows", "Value": f"{NOTEBOOK_METRICS['test_rows']:,}"},
            {"Field": "Observed churn rate", "Value": format_pct(dataset_summary.get("churn_rate"))},
        ]
        st.dataframe(pd.DataFrame(about_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Features used for inference**")
        st.caption("The saved model expects 22 cleaned-modeling features.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": MODEL_INPUT_COLUMNS,
                    "Description": [get_column_description(name) for name in MODEL_INPUT_COLUMNS],
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def render_pipeline_section(bundle: dict | None) -> None:
    st.markdown('<div class="section-title">Pipeline and Inference Flow</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">How preprocessing and prediction work in this project.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Preprocessing summary**")
        st.markdown("\n".join(f"- {line}" for line in PIPELINE_SUMMARY))
        if bundle is not None:
            st.markdown("**Artifact snapshot**")
            artifact_rows = [
                {"Field": "Artifact path", "Value": str(bundle["artifact_path"])},
                {"Field": "Pipeline type", "Value": bundle.get("pipeline_type")},
                {"Field": "Preprocessor", "Value": bundle.get("preprocessor_type")},
                {"Field": "Estimator", "Value": bundle.get("estimator_type")},
                {"Field": "Predict probabilities", "Value": bundle.get("supports_predict_proba")},
                {"Field": "Decision function", "Value": bundle.get("supports_decision_function")},
            ]
            st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Inference assumptions**")
        st.markdown("\n".join(f"- {line}" for line in INFERENCE_ASSUMPTIONS))
        st.markdown("**Manual input sections**")
        sections = [
            f"{section_name}: {', '.join(FRIENDLY_LABELS.get(col, col) for col in columns)}"
            for section_name, columns in FEATURE_GROUPS.items()
        ]
        st.markdown("\n".join(f"- {line}" for line in sections))
        st.markdown('</div>', unsafe_allow_html=True)


def select_input(column_name: str, default_value: str, reference_df: pd.DataFrame | None, key: str, disabled: bool = False) -> str:
    options = get_category_options(reference_df, column_name)
    if not options:
        options = KNOWN_CATEGORY_LEVELS[column_name]
    index = options.index(default_value) if default_value in options else 0
    return st.selectbox(
        FRIENDLY_LABELS[column_name],
        options=options,
        index=index,
        key=key,
        disabled=disabled,
        help=get_column_description(column_name),
    )


def numeric_input(column_name: str, default_value: float | int, key: str, disabled: bool = False) -> float | int:
    bounds = NUMERIC_BOUNDS[column_name]
    if isinstance(bounds["min"], int) and isinstance(bounds["max"], int):
        return int(
            st.number_input(
                FRIENDLY_LABELS[column_name],
                min_value=int(bounds["min"]),
                max_value=int(bounds["max"]),
                value=int(default_value),
                step=int(bounds["step"]),
                key=key,
                disabled=disabled,
                help=get_column_description(column_name),
            )
        )

    return float(
        st.number_input(
            FRIENDLY_LABELS[column_name],
            min_value=float(bounds["min"]),
            max_value=float(bounds["max"]),
            value=float(default_value),
            step=float(bounds["step"]),
            key=key,
            disabled=disabled,
            help=get_column_description(column_name),
        )
    )


def store_prediction_result(result: dict, source_label: str, intended_class: str | None = None) -> None:
    payload = dict(result)
    payload["source_label"] = source_label
    payload["intended_class"] = intended_class
    if intended_class is not None:
        payload["match_status"] = "Match" if result["predicted_label"] == intended_class else "Mismatch"
    else:
        payload["match_status"] = None
    st.session_state["latest_prediction"] = payload


def render_manual_input_tab(reference_df: pd.DataFrame | None, bundle: dict | None) -> None:
    defaults = build_manual_defaults(reference_df)
    customer_col, service_col, billing_col = st.columns(3)

    with customer_col:
        gender = select_input("gender", str(defaults["gender"]), reference_df, "manual_gender")
        seniorcitizen = select_input("seniorcitizen", str(defaults["seniorcitizen"]), reference_df, "manual_seniorcitizen")
        partner = select_input("partner", str(defaults["partner"]), reference_df, "manual_partner")
        dependents = select_input("dependents", str(defaults["dependents"]), reference_df, "manual_dependents")
        tenure = numeric_input("tenure", int(defaults["tenure"]), "manual_tenure")

    with service_col:
        phoneservice = select_input("phoneservice", str(defaults["phoneservice"]), reference_df, "manual_phoneservice")
        multiplelines_default = "unknown" if phoneservice == "No" else ("No" if defaults["multiplelines"] == "unknown" else str(defaults["multiplelines"]))
        multiple_options = ["unknown"] if phoneservice == "No" else ["No", "Yes"]
        multiplelines = st.selectbox(
            FRIENDLY_LABELS["multiplelines"],
            options=multiple_options,
            index=multiple_options.index(multiplelines_default) if multiplelines_default in multiple_options else 0,
            key="manual_multiplelines",
            disabled=phoneservice == "No",
            help=get_column_description("multiplelines"),
        )
        internetservice = select_input("internetservice", str(defaults["internetservice"]), reference_df, "manual_internetservice")
        contract = select_input("contract", str(defaults["contract"]), reference_df, "manual_contract")
        paperlessbilling = select_input("paperlessbilling", str(defaults["paperlessbilling"]), reference_df, "manual_paperless")

    with billing_col:
        paymentmethod = select_input("paymentmethod", str(defaults["paymentmethod"]), reference_df, "manual_paymentmethod")
        monthlycharges = numeric_input("monthlycharges", float(defaults["monthlycharges"]), "manual_monthlycharges")
        auto_totalcharges = st.toggle(
            "Estimate total charges from tenure × monthly charges",
            value=True,
            key="manual_auto_totalcharges",
            help="Recommended for quick testing. Turn this off if you want to enter total charges manually.",
        )
        estimated_totalcharges = round(tenure * monthlycharges, 2)
        totalcharges_default = estimated_totalcharges if auto_totalcharges else float(defaults["totalcharges"])
        totalcharges = numeric_input(
            "totalcharges",
            totalcharges_default,
            "manual_totalcharges",
            disabled=auto_totalcharges,
        )
        if auto_totalcharges:
            st.caption(f"Using an estimated total-charges value of `{estimated_totalcharges:,.2f}`.")

    st.markdown("**Add-on service selections**")
    if internetservice == "No":
        st.info("Internet service is set to `No`, so all internet add-on services are automatically forced to `No` for inference.")

    add_on_cols = st.columns(3)
    add_on_values: dict[str, str] = {}
    for index, column_name in enumerate(FEATURE_GROUPS["Add-on services"]):
        with add_on_cols[index % 3]:
            default_value = "No" if internetservice == "No" else str(defaults[column_name])
            add_on_values[column_name] = select_input(
                column_name,
                default_value,
                reference_df,
                f"manual_{column_name}",
                disabled=internetservice == "No",
            )

    base_inputs = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": "unknown" if phoneservice == "No" else multiplelines,
        "internetservice": internetservice,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": estimated_totalcharges if auto_totalcharges else totalcharges,
        **add_on_values,
    }

    preview_frame = build_inference_ready_preview(base_inputs)

    derived_cols = st.columns(3)
    derived_cols[0].metric("Customer loyalty band", str(preview_frame.loc[0, "cust-loyality"]))
    derived_cols[1].metric("Family segment", str(preview_frame.loc[0, "family_member"]))
    derived_cols[2].metric("Subscribed add-on count", int(preview_frame.loc[0, "subscription_count"]))

    st.markdown("**Exact feature row that will be sent to the pipeline**")
    st.dataframe(preview_frame, use_container_width=True, hide_index=True)

    if st.button(
        "Predict from manual input",
        type="primary",
        use_container_width=True,
        disabled=bundle is None,
        key="predict_manual",
    ):
        input_frame = build_inference_frame(base_inputs)
        result = predict_record(bundle, input_frame)
        store_prediction_result(result, source_label="Manual input")


def render_sample_generation_tab(reference_df: pd.DataFrame | None, bundle: dict | None) -> None:
    if reference_df is None:
        st.error("The reference dataset is unavailable, so auto-generated samples cannot be created.")
        return

    intended_class = st.radio(
        "Generate a historical sample intended for:",
        options=["No", "Yes"],
        horizontal=True,
        help="Samples are drawn from real rows in `cleaned_data.csv`.",
    )
    st.caption(TARGET_DESCRIPTIONS[intended_class])

    generate_col, predict_col = st.columns([1, 1])
    if generate_col.button("Generate sample profile", type="primary", use_container_width=True, key="generate_sample"):
        sample_counter = int(st.session_state.get("sample_counter", 0)) + 1
        st.session_state["sample_counter"] = sample_counter
        st.session_state["generated_sample"] = sample_reference_row(
            reference_df,
            intended_class=intended_class,
            random_state=sample_counter,
        )

    generated_sample = st.session_state.get("generated_sample")
    if generated_sample is None:
        st.info("Generate a class-targeted sample to preview it here.")
        return

    sample_header_cols = st.columns(3)
    sample_header_cols[0].metric("Intended class", generated_sample["intended_class"])
    sample_header_cols[1].metric("Source", "Historical profile")
    sample_header_cols[2].metric("Feature count", len(generated_sample["model_frame"].columns))

    st.markdown("**Generated sample preview**")
    st.dataframe(generated_sample["model_frame"], use_container_width=True, hide_index=True)

    if predict_col.button(
        "Predict generated sample",
        use_container_width=True,
        disabled=bundle is None,
        key="predict_sample",
    ):
        result = predict_record(bundle, generated_sample["model_frame"])
        store_prediction_result(
            result,
            source_label="Auto-generated sample",
            intended_class=generated_sample["intended_class"],
        )


def render_prediction_results() -> None:
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Latest prediction from either Manual Input mode or Auto-Generated Sample mode.</div>',
        unsafe_allow_html=True,
    )

    latest_prediction = st.session_state.get("latest_prediction")
    if latest_prediction is None:
        st.info("Run a prediction to populate the result panel.")
        return

    summary_method = st.error if latest_prediction["predicted_label"] == "Yes" else st.success
    summary_method(latest_prediction["summary_sentence"])

    metric_cols = st.columns(5)
    metric_cols[0].metric("Source", latest_prediction["source_label"])
    metric_cols[1].metric("Predicted class", latest_prediction["predicted_label"])
    metric_cols[2].metric("Churn probability", format_pct(latest_prediction.get("yes_probability")))
    metric_cols[3].metric("Model confidence", format_pct(latest_prediction.get("predicted_probability")))
    decision_value = latest_prediction.get("decision_score")
    metric_cols[4].metric("Decision score", "N/A" if decision_value is None else f"{decision_value:.4f}")

    if latest_prediction.get("intended_class") is not None:
        compare_cols = st.columns(3)
        compare_cols[0].metric("Intended class", latest_prediction["intended_class"])
        compare_cols[1].metric("Predicted class", latest_prediction["predicted_label"])
        compare_cols[2].metric("Match status", latest_prediction["match_status"])

    if latest_prediction["probabilities"]:
        probability_df = pd.DataFrame(
            {
                "Class": list(latest_prediction["probabilities"].keys()),
                "Probability": list(latest_prediction["probabilities"].values()),
            }
        )
        fig = px.bar(
            probability_df,
            x="Probability",
            y="Class",
            orientation="h",
            text_auto=".1%",
            title="Predicted class probabilities",
            color="Class",
            color_discrete_sequence=["#1d4ed8", "#f97316"],
        )
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=50, b=0), showlegend=False)
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Exact feature row used for prediction**")
    st.dataframe(latest_prediction["input_frame"], use_container_width=True, hide_index=True)


def render_insights(reference_df: pd.DataFrame | None, feature_importance_df: pd.DataFrame | None) -> None:
    st.markdown('<div class="section-title">Insights and Interpretation Notes</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Only grounded notes from the saved artifacts, the modeling notebook, and the cleaned project dataset.</div>',
        unsafe_allow_html=True,
    )

    insight_cards = build_insight_cards(reference_df)
    if insight_cards:
        card_cols = st.columns(len(insight_cards))
        for index, card in enumerate(insight_cards):
            with card_cols[index]:
                st.metric(card["title"], card["label"], delta=f"{card['value']:.1%} churn rate")

    left_col, right_col = st.columns([1.05, 0.95])
    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Most influential transformed features**")
        st.caption("These values come from `artifacts/model_features.csv` and reflect transformed-model feature importance.")
        if feature_importance_df is None or feature_importance_df.empty:
            st.warning("Feature-importance artifact could not be loaded.")
        else:
            top_features = feature_importance_df.head(10).sort_values("importance", ascending=True)
            fig = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                text_auto=".2f",
                color="importance",
                color_continuous_scale=["#cbd5e1", "#1d4ed8"],
            )
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        dimension = st.selectbox(
            "Review descriptive churn-rate patterns by:",
            options=["contract", "internetservice", "paymentmethod", "cust-loyality"],
            key="insight_dimension",
        )
        rate_table = build_rate_table(reference_df, dimension)
        if rate_table.empty:
            st.warning("No descriptive rate table is available for this view.")
        else:
            fig = px.bar(
                rate_table,
                x=dimension,
                y="churn_rate",
                text_auto=".1%",
                color="churn_rate",
                color_continuous_scale=["#dbeafe", "#f97316"],
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rate_table, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    notes_col, caveat_col = st.columns(2)
    with notes_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Usage notes**")
        st.markdown("\n".join(f"- {item}" for item in USAGE_NOTES))
        st.markdown('</div>', unsafe_allow_html=True)
    with caveat_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Limitations and caveats**")
        st.markdown("\n".join(f"- {item}" for item in LIMITATIONS))
        st.markdown('</div>', unsafe_allow_html=True)


def render_debug_section(bundle: dict | None, bundle_error: Exception | None) -> None:
    st.markdown('<div class="section-title">Debug and Transparency</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A compact place to inspect paths, schema expectations, and the exact pipeline contract.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Open debug details"):
        st.markdown("**Artifact paths**")
        st.code(str(DEFAULT_MODEL_ARTIFACT), language="text")
        st.code(str(DEFAULT_FEATURE_IMPORTANCE_ARTIFACT), language="text")
        st.code(str(DEFAULT_REFERENCE_DATASET), language="text")

        if bundle is not None:
            st.markdown("**Saved bundle summary**")
            st.json(build_debug_summary(bundle))
        elif bundle_error is not None:
            st.error(str(bundle_error))

        st.markdown("**Expected raw input schema**")
        st.write(MODEL_INPUT_COLUMNS)

        latest_prediction = st.session_state.get("latest_prediction")
        if latest_prediction is not None:
            st.markdown("**Latest schema warnings**")
            warnings = latest_prediction.get("schema_warnings") or []
            if warnings:
                st.warning("\n".join(warnings))
            else:
                st.success("No schema issues were detected for the latest prediction row.")
            st.markdown("**Latest inference row preview**")
            st.dataframe(latest_prediction["input_frame"], use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()

    bundle = None
    bundle_error: Exception | None = None
    try:
        bundle = get_bundle()
    except (ArtifactLoadError, FileNotFoundError, ValueError) as exc:
        bundle_error = exc
    except Exception as exc:
        bundle_error = exc

    reference_df = None
    reference_error: Exception | None = None
    try:
        reference_df = get_reference_data()
    except Exception as exc:
        reference_error = exc

    feature_importance_df = None
    feature_importance_error: Exception | None = None
    try:
        feature_importance_df = get_feature_importance_data()
    except Exception as exc:
        feature_importance_error = exc

    dataset_summary = build_dataset_summary(reference_df)

    render_sidebar(bundle, dataset_summary, bundle_error)
    render_hero(bundle, dataset_summary)

    if reference_error is not None:
        st.warning(f"Reference dataset note: {reference_error}")
    if feature_importance_error is not None:
        st.warning(f"Feature-importance note: {feature_importance_error}")
    if bundle_error is not None:
        st.warning(
            "The informational sections below are still available, but live inference is disabled until the saved model artifact can be loaded."
        )

    render_about_model(bundle, dataset_summary)
    render_pipeline_section(bundle)

    st.markdown('<div class="section-title">Interactive Testing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Test the saved churn model manually or with class-targeted historical samples.</div>',
        unsafe_allow_html=True,
    )
    manual_tab, sample_tab = st.tabs(["Manual Input", "Auto-Generated Sample"])
    with manual_tab:
        render_manual_input_tab(reference_df, bundle)
    with sample_tab:
        render_sample_generation_tab(reference_df, bundle)

    render_prediction_results()
    render_insights(reference_df, feature_importance_df)
    render_debug_section(bundle, bundle_error)


if __name__ == "__main__":
    main()
