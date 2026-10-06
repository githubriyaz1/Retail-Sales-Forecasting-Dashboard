import os
import pandas as pd


# ============================================================
# RETAIL SALES DATA CLEANING & PREPROCESSING
# ============================================================

INPUT_FILE = "data/raw/retail_sales_raw.csv"
OUTPUT_FILE = "data/processed/retail_sales_clean.csv"


def load_data():
    """Load the raw retail sales dataset."""
    df = pd.read_csv(INPUT_FILE)

    print(f"Raw dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    return df


def clean_data(df):
    """Clean and preprocess the retail sales dataset."""

    print("\nStarting data cleaning...")

    # --------------------------------------------------------
    # 1. Convert Order_Date to datetime
    # --------------------------------------------------------

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 2. Remove duplicate records
    # --------------------------------------------------------

    duplicates = df.duplicated().sum()

    print(f"Duplicate records found: {duplicates}")

    df = df.drop_duplicates().copy()

    # --------------------------------------------------------
    # 3. Handle missing values
    # --------------------------------------------------------

    missing_before = df.isnull().sum().sum()

    print(f"Missing values before cleaning: {missing_before}")

    # Remove rows with missing critical information
    df = df.dropna(
        subset=[
            "Order_ID",
            "Order_Date",
            "Product_ID",
            "Product_Name",
            "Category",
            "Region",
            "Quantity",
            "Revenue",
            "Profit",
        ]
    )

    missing_after = df.isnull().sum().sum()

    print(f"Missing values after cleaning: {missing_after}")

    # --------------------------------------------------------
    # 4. Standardize text columns
    # --------------------------------------------------------

    text_columns = [
        "Product_Name",
        "Category",
        "Region",
        "Customer_Type",
        "Payment_Method",
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    # --------------------------------------------------------
    # 5. Ensure numerical columns are numeric
    # --------------------------------------------------------

    numeric_columns = [
        "Quantity",
        "Unit_Price",
        "Discount",
        "Revenue",
        "Cost",
        "Profit",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 6. Remove invalid numerical records
    # --------------------------------------------------------

    df = df[
        (df["Quantity"] > 0)
        & (df["Unit_Price"] > 0)
        & (df["Revenue"] > 0)
        & (df["Cost"] >= 0)
        & (df["Profit"] >= 0)
        & (df["Discount"] >= 0)
        & (df["Discount"] <= 100)
    ].copy()

    # --------------------------------------------------------
    # 7. Create useful analytical columns
    # --------------------------------------------------------

    df["Year"] = df["Order_Date"].dt.year

    df["Month"] = df["Order_Date"].dt.month

    df["Month_Name"] = df["Order_Date"].dt.strftime("%B")

    df["Quarter"] = (
        "Q" + df["Order_Date"].dt.quarter.astype(str)
    )

    df["Day"] = df["Order_Date"].dt.day

    df["Day_Name"] = df["Order_Date"].dt.strftime("%A")

    # --------------------------------------------------------
    # 8. Calculate profit margin
    # --------------------------------------------------------

    df["Profit_Margin"] = (
        df["Profit"] / df["Revenue"]
    ) * 100

    # --------------------------------------------------------
    # 9. Calculate discount amount
    # --------------------------------------------------------

    df["Discount_Amount"] = (
        df["Quantity"]
        * df["Unit_Price"]
        * df["Discount"]
        / 100
    )

    # --------------------------------------------------------
    # 10. Round numerical values
    # --------------------------------------------------------

    decimal_columns = [
        "Unit_Price",
        "Revenue",
        "Cost",
        "Profit",
        "Profit_Margin",
        "Discount_Amount",
    ]

    for column in decimal_columns:
        df[column] = df[column].round(2)

    # --------------------------------------------------------
    # 11. Sort by date
    # --------------------------------------------------------

    df = df.sort_values(
        "Order_Date"
    ).reset_index(drop=True)

    return df


def save_data(df):
    """Save the cleaned dataset."""

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nCleaned dataset saved to: {OUTPUT_FILE}"
    )


def main():

    print("=" * 60)
    print("RETAIL SALES DATA CLEANING")
    print("=" * 60)

    df = load_data()

    df = clean_data(df)

    save_data(df)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nFinal rows    : {len(df):,}")
    print(f"Final columns : {len(df.columns)}")

    print("\nFinal columns:")

    for column in df.columns:
        print(f" - {column}")

    print("\nFinal dataset preview:")
    print(df.head())

    print("\nFinal missing values:")

    print(
        df.isnull().sum()
    )


if __name__ == "__main__":
    main()