"""
Marikyan - Python Data Science Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ─────────────────────────────────────────────
# 1. Load / Generate Data
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Replace this with your own data source, e.g.:
        pd.read_csv("data/myfile.csv")
        pd.read_excel("data/myfile.xlsx")
    """
    rng = np.random.default_rng(seed=42)
    n = 200

    X = rng.uniform(0, 10, n)
    noise = rng.normal(0, 1.5, n)
    y = 3.5 * X + 7.0 + noise

    df = pd.DataFrame({"feature": X, "target": y})
    return df


# ─────────────────────────────────────────────
# 2. Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────

def eda(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("Dataset shape :", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nDescriptive statistics:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    print("=" * 50)


# ─────────────────────────────────────────────
# 3. Visualise
# ─────────────────────────────────────────────

def plot_data(df: pd.DataFrame, save_path: str = "output/scatter.png") -> None:
    import os
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    axes[0].scatter(df["feature"], df["target"], alpha=0.6, color="steelblue")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Target")
    axes[0].set_title("Feature vs Target")

    # Distribution
    axes[1].hist(df["target"], bins=20, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Target value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Target Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved → {save_path}")


# ─────────────────────────────────────────────
# 4. Model Training & Evaluation
# ─────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> None:
    X = df[["feature"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Results")
    print("-" * 30)
    print(f"  Coefficient : {model.coef_[0]:.4f}")
    print(f"  Intercept   : {model.intercept_:.4f}")
    print(f"  MSE         : {mse:.4f}")
    print(f"  R² Score    : {r2:.4f}")


# ─────────────────────────────────────────────
# 5. Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    eda(df)
    plot_data(df)
    train_and_evaluate(df)
