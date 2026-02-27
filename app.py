import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score
)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# ======================================
# ML FUNCTIONS
# ======================================

def run_classification(df, target, features, test_size):

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, cm


def run_regression(df, target, features, test_size):

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


# ======================================
# STREAMLIT UI
# ======================================

st.title("🧠 Machine Learning & Analytics Dashboard")

uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

# =========================================================
# AFTER FILE UPLOAD → SHOW NAVIGATION
# =========================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Choose mode
    mode = st.radio(
        "🔎 Choose Section",
        ["📊 Exploratory Analytics", "🧠 Models"]
    )

    st.markdown("---")

    # =====================================================
    # 📊 EXPLORATORY ANALYTICS
    # =====================================================
    if mode == "📊 Exploratory Analytics":

        st.header("📊 Exploratory Data Analysis")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Dataset Shape")
            st.write(df.shape)

            st.write("### Data Types")
            st.write(df.dtypes)

        with col2:
            st.write("### Missing Values")
            st.write(df.isnull().sum())

        st.write("### Statistical Summary")
        st.dataframe(df.describe())

        # Correlation Heatmap
        st.write("### Correlation Heatmap")

        numeric_df = df.select_dtypes(include=np.number)

        if not numeric_df.empty:
            fig, ax = plt.subplots()
            cax = ax.imshow(numeric_df.corr(), cmap="coolwarm")
            plt.colorbar(cax)
            ax.set_title("Correlation Matrix")

            st.pyplot(fig)
        else:
            st.info("No numeric columns available.")

        # Target distribution
        target_for_plot = st.selectbox(
            "Select column for distribution",
            df.columns
        )

        st.write("### Distribution")

        fig2, ax2 = plt.subplots()
        df[target_for_plot].value_counts().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    # =====================================================
    # 🧠 MODELS SECTION
    # =====================================================
    else:

        st.header("🧠 Machine Learning Models")

        problem_type = st.radio(
            "🔍 Select Problem Type",
            ["Classification", "Regression"]
        )

        # Default target logic
        default_index = 0
        if problem_type == "Classification" and "species" in df.columns:
            default_index = list(df.columns).index("species")

        target = st.selectbox(
            "🎯 Select Target Variable",
            df.columns,
            index=default_index
        )

        features = st.multiselect(
            "📊 Select Feature Columns",
            [col for col in df.columns if col != target]
        )

        test_size = st.slider(
            "⚖️ Train-Test Split",
            0.1, 0.5, 0.2
        )

        if st.button("🚀 Evaluate Model"):

            if not features:
                st.error("Please select at least one feature.")

            else:

                if problem_type == "Classification":

                    acc, cm = run_classification(
                        df, target, features, test_size
                    )

                    st.success(f"✅ Accuracy: {acc:.4f}")

                    st.subheader("Confusion Matrix")
                    st.write(cm)

                else:

                    mse, r2 = run_regression(
                        df, target, features, test_size
                    )

                    st.success(f"📉 MSE: {mse:.4f}")
                    st.success(f"📈 R² Score: {r2:.4f}")

else:
    st.info("👆 Upload a CSV file to begin.")