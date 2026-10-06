from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# RETAIL SALES - FINAL FORECASTING ENGINE
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
# Load Data
# ------------------------------------------------------------

def load_data():

    df = pd.read_csv(
        INPUT_FILE
    )

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"]
    )

    return df


# ------------------------------------------------------------
# Prepare Monthly Revenue
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

    data["Time_Index"] = np.arange(
        len(data)
    )

    data["Month_Number"] = (
        data["Month"].dt.month
    )

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

    data["Lag_1"] = (
        data["Revenue"]
        .shift(1)
    )

    data["Lag_2"] = (
        data["Revenue"]
        .shift(2)
    )

    data["Lag_3"] = (
        data["Revenue"]
        .shift(3)
    )

    data["Lag_6"] = (
        data["Revenue"]
        .shift(6)
    )

    data["Rolling_3"] = (
        data["Revenue"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    data["Rolling_6"] = (
        data["Revenue"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    return data


# ------------------------------------------------------------
# Create Model
# ------------------------------------------------------------

def create_model():

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )

    return model


# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

def train_model():

    df = load_data()

    monthly = prepare_monthly_data(
        df
    )

    data = create_features(
        monthly
    )

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    X = data[
        FEATURE_COLUMNS
    ]

    y = data["Revenue"]

    model = create_model()

    model.fit(
        X,
        y
    )

    return model, monthly


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

def evaluate_model():

    df = load_data()

    monthly = prepare_monthly_data(
        df
    )

    data = create_features(
        monthly
    )

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    X = data[
        FEATURE_COLUMNS
    ]

    y = data["Revenue"]

    split_index = int(
        len(data) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

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
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


# ------------------------------------------------------------
# Recursive Future Forecast
# ------------------------------------------------------------

def forecast_future(
    model,
    monthly,
    periods=6
):

    history = (
        monthly[
            ["Month", "Revenue"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    predictions = []

    for _ in range(periods):

        next_month = (
            history["Month"].max()
            + pd.DateOffset(
                months=1
            )
        )

        time_index = len(history)

        month_number = (
            next_month.month
        )

        month_sin = np.sin(
            2
            * np.pi
            * month_number
            / 12
        )

        month_cos = np.cos(
            2
            * np.pi
            * month_number
            / 12
        )

        lag_1 = (
            history["Revenue"]
            .iloc[-1]
        )

        lag_2 = (
            history["Revenue"]
            .iloc[-2]
        )

        lag_3 = (
            history["Revenue"]
            .iloc[-3]
        )

        lag_6 = (
            history["Revenue"]
            .iloc[-6]
        )

        rolling_3 = (
            history["Revenue"]
            .iloc[-3:]
            .mean()
        )

        rolling_6 = (
            history["Revenue"]
            .iloc[-6:]
            .mean()
        )

        features = pd.DataFrame(
            {
                "Time_Index": [
                    time_index
                ],

                "Month_Sin": [
                    month_sin
                ],

                "Month_Cos": [
                    month_cos
                ],

                "Lag_1": [
                    lag_1
                ],

                "Lag_2": [
                    lag_2
                ],

                "Lag_3": [
                    lag_3
                ],

                "Lag_6": [
                    lag_6
                ],

                "Rolling_3": [
                    rolling_3
                ],

                "Rolling_6": [
                    rolling_6
                ],
            }
        )

        prediction = model.predict(
            features[
                FEATURE_COLUMNS
            ]
        )[0]

        prediction = max(
            0,
            prediction
        )

        prediction = round(
            prediction,
            2
        )

        predictions.append(
            {
                "Month": next_month,
                "Predicted_Revenue": prediction
            }
        )

        # Add prediction to history
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "Month": [
                            next_month
                        ],
                        "Revenue": [
                            prediction
                        ],
                    }
                )
            ],
            ignore_index=True
        )

    return pd.DataFrame(
        predictions
    )


# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

def save_model(model):

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"Model saved to:\n{MODEL_FILE}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("FINAL RETAIL SALES FORECASTING ENGINE")
    print("=" * 70)

    # Evaluate
    metrics = evaluate_model()

    print("\nMODEL PERFORMANCE")
    print("-" * 70)

    print(
        f"MAE  : ₹{metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE : ₹{metrics['RMSE']:,.2f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print(
        f"MAPE : {metrics['MAPE']:.2f}%"
    )

    # Train on complete historical dataset
    model, monthly = train_model()

    # Save model
    save_model(
        model
    )

    # Forecast
    future = forecast_future(
        model,
        monthly,
        periods=6
    )

    print("\nNEXT 6 MONTH SALES FORECAST")
    print("-" * 70)

    display_forecast = future.copy()

    display_forecast["Month"] = (
        display_forecast["Month"]
        .dt.strftime("%B %Y")
    )

    display_forecast[
        "Predicted_Revenue"
    ] = (
        display_forecast[
            "Predicted_Revenue"
        ]
        .apply(
            lambda x:
            f"₹{x:,.2f}"
        )
    )

    print(
        display_forecast.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FINAL FORECASTING ENGINE READY")
    print("=" * 70)


if __name__ == "__main__":
    main()