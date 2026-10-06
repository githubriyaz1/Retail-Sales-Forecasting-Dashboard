from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# RETAIL SALES MODEL COMPARISON
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_sales_clean.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_sales_forecasting_model.pkl"
)


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def load_data():

    df = pd.read_csv(INPUT_FILE)

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"]
    )

    return df


# ------------------------------------------------------------
# Monthly Data
# ------------------------------------------------------------

def prepare_monthly_data(df):

    monthly = (
        df.groupby(
            df["Order_Date"].dt.to_period("M")
        )["Revenue"]
        .sum()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Revenue"
    ]

    monthly["Month"] = (
        monthly["Month"]
        .dt.to_timestamp()
    )

    monthly = (
        monthly
        .sort_values("Month")
        .reset_index(drop=True)
    )

    return monthly


# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

def create_features(monthly):

    data = monthly.copy()

    # Time trend
    data["Time_Index"] = np.arange(
        len(data)
    )

    # Month
    data["Month_Number"] = (
        data["Month"].dt.month
    )

    # Seasonal features
    data["Month_Sin"] = np.sin(
        2
        * np.pi
        * data["Month_Number"]
        / 12
    )

    data["Month_Cos"] = np.cos(
        2
        * np.pi
        * data["Month_Number"]
        / 12
    )

    # Lag features
    data["Lag_1"] = (
        data["Revenue"].shift(1)
    )

    data["Lag_2"] = (
        data["Revenue"].shift(2)
    )

    data["Lag_3"] = (
        data["Revenue"].shift(3)
    )

    data["Lag_6"] = (
        data["Revenue"].shift(6)
    )

    # 3-month rolling average
    data["Rolling_3"] = (
        data["Revenue"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    # 6-month rolling average
    data["Rolling_6"] = (
        data["Revenue"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    return data


# ------------------------------------------------------------
# Feature Columns
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "Time_Index",
    "Month_Sin",
    "Month_Cos",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Rolling_3",
    "Rolling_6",
]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(
    y_test,
    predictions
):

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    non_zero = y_test != 0

    mape = (
        np.mean(
            np.abs(
                (
                    y_test[non_zero]
                    - predictions[non_zero]
                )
                / y_test[non_zero]
            )
        )
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


# ------------------------------------------------------------
# Train and Evaluate Model
# ------------------------------------------------------------

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return {
        "Model": name,
        **metrics,
        "Trained_Model": model,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 75)
    print("RETAIL SALES FORECASTING - MODEL COMPARISON")
    print("=" * 75)

    # Load data
    df = load_data()

    # Monthly aggregation
    monthly = prepare_monthly_data(
        df
    )

    print(
        f"\nHistorical months: {len(monthly)}"
    )

    # Feature engineering
    data = create_features(
        monthly
    )

    # Remove rows created by lag features
    data = data.dropna().reset_index(
        drop=True
    )

    print(
        f"Usable ML records: {len(data)}"
    )

    # Features and target
    X = data[FEATURE_COLUMNS]

    y = data["Revenue"]

    # Chronological split
    split_index = int(
        len(data) * 0.80
    )

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    print(
        f"Training records: {len(X_train)}"
    )

    print(
        f"Testing records : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = [

        (
            "Linear Regression",
            LinearRegression()
        ),

        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                random_state=42
            )
        ),

        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                random_state=42
            )
        ),
    ]

    results = []

    # --------------------------------------------------------
    # Train Models
    # --------------------------------------------------------

    for name, model in models:

        print(
            f"\nTraining: {name}"
        )

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

    # --------------------------------------------------------
    # Results Table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            {
                "Model": result["Model"],
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "R2": result["R2"],
                "MAPE": result["MAPE"],
            }
            for result in results
        ]
    )

    results_df = results_df.sort_values(
        "MAPE"
    ).reset_index(drop=True)

    print("\n")
    print("=" * 75)
    print("MODEL COMPARISON RESULTS")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE": "{:,.2f}".format,
                "RMSE": "{:,.2f}".format,
                "R2": "{:.4f}".format,
                "MAPE": "{:.2f}%".format,
            }
        )
    )

    # --------------------------------------------------------
    # Select Best Model
    # --------------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_result = next(
        result
        for result in results
        if result["Model"] == best_model_name
    )

    best_model = (
        best_result["Trained_Model"]
    )

    print("\n")
    print("=" * 75)
    print("BEST MODEL")
    print("=" * 75)

    print(
        f"Selected Model: {best_model_name}"
    )

    print(
        f"MAPE: {best_result['MAPE']:.2f}%"
    )

    print(
        f"R²: {best_result['R2']:.4f}"
    )

    # --------------------------------------------------------
    # Save Best Model
    # --------------------------------------------------------

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_FILE
    )

    print(
        f"\nBest model saved to:"
    )

    print(
        MODEL_FILE
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()