
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================
# AIRGUARD - AIR QUALITY CLASSIFICATION SYSTEM
# ============================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3"
]

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
    "Neural Network (MLP)": "neural_network.joblib"
}

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AirGuard - Air Quality AI",
    page_icon="🌿",
    layout="wide"
)


# ============================================
# DESIGN
# ============================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1f27;
    color: #f0f7f5;
}

[data-testid="stSidebar"] {
    background-color: #102d37;
}

h1, h2, h3 {
    color: #55d6ad !important;
}

p, label {
    color: #d5e5e2;
}

div[data-testid="stMetric"] {
    background: #153944;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #285560;
}

.stButton > button {
    background-color: #079b80;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #10b99c;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODELS AND METRICS
# ============================================

@st.cache_resource
def load_project():

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        path = MODEL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing model file: {filename}"
            )

        loaded_models[name] = joblib.load(path)

    metrics_path = MODEL_DIR / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(
            "models/metrics.json was not found."
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return loaded_models, data


try:
    models, project_data = load_project()

except Exception as error:
    st.error(f"Project loading error: {error}")
    st.info(
        "Check that the models folder contains all three "
        "trained model files and metrics.json."
    )
    st.stop()


# ============================================
# READ METRICS JSON CORRECTLY
# ============================================

# Your metrics.json stores model results under "metrics".
# Defaults and classes are stored at the top level.

model_metrics_data = project_data.get("metrics", {})

if not isinstance(model_metrics_data, dict):
    st.error("The metrics section in metrics.json is invalid.")
    st.stop()


def normalize_name(name):

    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def find_model_metrics(model_name):

    aliases = {
        "Random Forest": [
            "randomforest",
            "randomforestclassifier",
            "rf"
        ],

        "Support Vector Machine (SVM)": [
            "svm",
            "supportvectormachine",
            "supportvectorclassifier"
        ],

        "Neural Network (MLP)": [
            "neuralnetwork",
            "mlp",
            "mlpclassifier",
            "neuralnetworkclassifier"
        ]
    }

    candidates = [
        normalize_name(model_name)
    ] + aliases[model_name]

    for saved_name, saved_metrics in model_metrics_data.items():

        normalized_saved_name = normalize_name(saved_name)

        if normalized_saved_name in candidates:
            return saved_metrics

    # Fallback: match model names by their identifying words.
    for saved_name, saved_metrics in model_metrics_data.items():

        key = normalize_name(saved_name)

        if model_name == "Random Forest" and "randomforest" in key:
            return saved_metrics

        if model_name == "Support Vector Machine (SVM)" and (
            "svm" in key or "supportvector" in key
        ):
            return saved_metrics

        if model_name == "Neural Network (MLP)" and (
            "neuralnetwork" in key or "mlp" in key
        ):
            return saved_metrics

    raise KeyError(
        f"Cannot match {model_name}. "
        f"Saved model names: {list(model_metrics_data.keys())}"
    )


def get_metric(model_name, metric_name):

    saved = find_model_metrics(model_name)

    aliases = {
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision", "weighted_precision"],
        "recall": ["recall", "weighted_recall"],
        "f1": ["f1", "f1_score", "weighted_f1"]
    }

    for key, value in saved.items():

        if normalize_name(key) in [
            normalize_name(item)
            for item in aliases[metric_name]
        ]:
            return float(value)

    raise KeyError(
        f"{metric_name} not found for {model_name}. "
        f"Available metrics: {list(saved.keys())}"
    )


def to_percentage(value):

    if value <= 1:
        return value * 100

    return value


def get_best_model():

    best_name = max(
        MODEL_FILES.keys(),
        key=lambda name: get_metric(name, "accuracy")
    )

    accuracy = to_percentage(
        get_metric(best_name, "accuracy")
    )

    return best_name, accuracy


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:

    st.title("🌿 AirGuard")

    st.caption("AI-Powered Air Quality System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Predict Air Quality",
            "📊 Model Comparison",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### Algorithms")

    st.write("🌳 Random Forest")
    st.write("🎯 Support Vector Machine")
    st.write("🧠 Neural Network")

    st.divider()

    st.caption("Machine Learning Academic Project")


# ============================================
# HEADER
# ============================================

st.title("🌿 AirGuard")

st.write(
    "An intelligent Air Quality Classification System "
    "powered by Machine Learning."
)

st.markdown("---")


# ============================================
# DASHBOARD
# ============================================

if page == "🏠 Dashboard":

    st.header("Project Dashboard")

    best_model, best_accuracy = get_best_model()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Machine Learning Algorithms",
        "03"
    )

    col2.metric(
        "Best Performing Model",
        best_model
    )

    col3.metric(
        "Best Accuracy",
        f"{best_accuracy:.2f}%"
    )

    st.markdown("---")

    st.subheader("Welcome to AirGuard")

    st.write("""
    AirGuard is a Machine Learning based Air Quality
    Classification System developed using historical
    air pollution data from India.

    The system analyses pollutant concentrations and
    predicts one of three air quality categories:

    🌿 Good

    🌤️ Moderate

    ⚠️ Poor
    """)

    st.subheader("Algorithms Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🌳 Random Forest")

    with col2:
        st.info("🎯 Support Vector Machine")

    with col3:
        st.info("🧠 Neural Network")

    st.markdown("---")

    st.subheader("Model Accuracy Comparison")

    accuracy_data = []

    for name in MODEL_FILES:

        accuracy_data.append({
            "Algorithm": name,
            "Accuracy": to_percentage(
                get_metric(name, "accuracy")
            )
        })

    accuracy_df = pd.DataFrame(accuracy_data)

    fig = px.bar(
        accuracy_df,
        x="Algorithm",
        y="Accuracy",
        color="Algorithm",
        text_auto=".2f",
        title="Accuracy of All Three Algorithms"
    )

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        yaxis_title="Accuracy (%)"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================
# PREDICT AIR QUALITY
# ============================================

elif page == "🔍 Predict Air Quality":

    st.header("🔍 Air Quality Prediction")

    st.write("""
    Enter the pollutant concentration values below.
    Select an algorithm to predict the air quality category.
    """)

    selected_model = st.selectbox(
        "Select Machine Learning Algorithm",
        list(MODEL_FILES.keys())
    )

    st.markdown("---")

    st.subheader("Pollutant Concentrations")

    # Correct defaults location from metrics.json
    defaults = project_data.get("defaults", {})

    # Also support a nested defaults dictionary if present
    if not isinstance(defaults, dict):
        defaults = {}

    input_values = {}

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for index, feature in enumerate(FEATURES):

        default_value = float(
            defaults.get(feature, 0.0)
        )

        with columns[index % 3]:

            input_values[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=max(0.0, default_value),
                step=0.1,
                format="%.2f"
            )

    st.markdown("---")

    if st.button(
        "🌿 Predict Air Quality",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        model = models[selected_model]

        prediction = str(model.predict(input_df)[0])

        st.subheader("Prediction Result")

        if prediction == "Good":

            st.success(
                "🌿 GOOD - The predicted category is Good."
            )

        elif prediction == "Moderate":

            st.warning(
                "🌤️ MODERATE - The predicted category is Moderate."
            )

        elif prediction == "Poor":

            st.error(
                "⚠️ POOR - The predicted category is Poor."
            )

        else:

            st.info(f"Predicted category: {prediction}")

        st.write("Selected algorithm:", selected_model)

        # Probability chart
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            if hasattr(model, "named_steps"):
                classes = model.named_steps[
                    "classifier"
                ].classes_
            else:
                classes = model.classes_

            probability_df = pd.DataFrame({
                "Category": classes,
                "Probability (%)": probabilities * 100
            })

            st.subheader("Prediction Probability")

            fig = px.bar(
                probability_df,
                x="Category",
                y="Probability (%)",
                color="Category",
                text_auto=".2f",
                title="Probability for Each Category"
            )

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.caption(
            "Academic prediction using historical data. "
            "This is not a live official AQI measurement "
            "or medical recommendation."
        )


# ============================================
# MODEL COMPARISON
# ============================================

elif page == "📊 Model Comparison":

    st.header("📊 Model Performance Comparison")

    st.write(
        "Compare the three algorithms using standard "
        "classification evaluation metrics."
    )

    comparison_data = []

    for name in MODEL_FILES:

        comparison_data.append({

            "Algorithm": name,

            "Accuracy (%)": to_percentage(
                get_metric(name, "accuracy")
            ),

            "Precision (%)": to_percentage(
                get_metric(name, "precision")
            ),

            "Recall (%)": to_percentage(
                get_metric(name, "recall")
            ),

            "F1 Score (%)": to_percentage(
                get_metric(name, "f1")
            )
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.subheader("Performance Metrics Table")

    st.dataframe(
        comparison_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    selected_metric = st.selectbox(
        "Choose metric for comparison",
        [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)"
        ]
    )

    fig = px.bar(
        comparison_df,
        x="Algorithm",
        y=selected_metric,
        color="Algorithm",
        text_auto=".2f",
        title=f"Comparison using {selected_metric}"
    )

    fig.update_layout(
        template="plotly_dark",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Confusion Matrices")

    # Classes are saved at the top level of metrics.json
    class_labels = project_data.get(
        "classes",
        ["Good", "Moderate", "Poor"]
    )

    for name in MODEL_FILES:

        with st.expander(f"View {name} Confusion Matrix"):

            saved = find_model_metrics(name)

            cm = saved.get("confusion_matrix")

            if cm is not None:

                cm_df = pd.DataFrame(
                    np.array(cm),
                    index=class_labels,
                    columns=class_labels
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                fig = px.imshow(
                    cm_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Teal",
                    title=f"{name} Confusion Matrix"
                )

                fig.update_layout(
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Confusion matrix is not available "
                    "in the saved metrics."
                )

    best_model, best_accuracy = get_best_model()

    st.success(
        f"Best performing model: {best_model} "
        f"({best_accuracy:.2f}% accuracy)"
    )


# ============================================
# ABOUT PROJECT
# ============================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About AirGuard")

    st.write("""
    AirGuard is an academic Machine Learning project
    that classifies air quality using pollutant
    concentration measurements.

    It uses historical air quality data from India
    to train and evaluate three classification algorithms.
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    1. Analyse historical air pollution data.
    2. Train different Machine Learning algorithms.
    3. Predict air quality categories.
    4. Compare classification performance.
    5. Develop an interactive website.
    """)

    st.markdown("---")

    st.subheader("Machine Learning Algorithms")

    with st.expander("🌳 Random Forest"):
        st.write(
            "An ensemble algorithm that combines multiple "
            "decision trees to improve classification."
        )

    with st.expander("🎯 Support Vector Machine"):
        st.write(
            "A supervised learning algorithm that identifies "
            "decision boundaries between different classes."
        )

    with st.expander("🧠 Neural Network"):
        st.write(
            "A multilayer perceptron that learns patterns "
            "through interconnected computational units."
        )

    st.markdown("---")

    st.subheader("Input Features")

    st.write(", ".join(FEATURES))

    st.subheader("Output Categories")

    st.write("🌿 Good | 🌤️ Moderate | ⚠️ Poor")

    st.subheader("Dataset")

    st.markdown(
        "[India Air Quality Dataset - Kaggle]"
        "(https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)"
    )

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib
    """)

    st.warning("""
    Disclaimer: This project is an academic demonstration
    using historical data. It does not provide live official
    AQI measurements, official CPCB classifications, or
    medical advice.
    """)


# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.caption(
    "🌿 AirGuard | Air Quality Classification System | "
    "Machine Learning Academic Project"
)
