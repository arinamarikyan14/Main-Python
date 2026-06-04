import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# ── 1. Load / generate data ──────────────────────────────────────────────────

def load_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, n)
    noise = rng.normal(0, 1, n)
    y = 2.5 * X + 1.0 + noise
    return pd.DataFrame({"feature": X, "target": y})


# ── 2. Exploratory data analysis ─────────────────────────────────────────────

def explore(df: pd.DataFrame) -> None:
    print("=== Shape ===")
    print(df.shape)
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Descriptive statistics ===")
    print(df.describe())
    print("\n=== Missing values ===")
    print(df.isnull().sum())


# ── 3. Visualisation ─────────────────────────────────────────────────────────

def visualise(df: pd.DataFrame, out_dir: str = "output") -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="feature", y="target", alpha=0.6, ax=ax)
    ax.set_title("Feature vs Target")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    path = os.path.join(out_dir, "scatter.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {path}")


# ── 4. Model ─────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> None:
    X = df[["feature"]].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n=== Linear Regression ===")
    print(f"  Coef : {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  R²   : {r2_score(y_test, preds):.4f}")
    print(f"  RMSE : {mean_squared_error(y_test, preds) ** 0.5:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    explore(df)
    visualise(df)
    train_model(df)
