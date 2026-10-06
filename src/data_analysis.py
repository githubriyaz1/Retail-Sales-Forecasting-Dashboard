import pandas as pd


# ============================================================
# RETAIL SALES BUSINESS INTELLIGENCE ANALYSIS
# ============================================================

INPUT_FILE = "data/processed/retail_sales_clean.csv"


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def load_data():
    """Load the cleaned retail sales dataset."""

    df = pd.read_csv(INPUT_FILE)

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"]
    )

    return df


# ------------------------------------------------------------
# KPI Calculations
# ------------------------------------------------------------

def calculate_kpis(df):
    """Calculate the main business KPIs."""

    total_revenue = df["Revenue"].sum()

    total_profit = df["Profit"].sum()

    total_orders = df["Order_ID"].nunique()

    total_units = df["Quantity"].sum()

    average_order_value = (
        total_revenue / total_orders
    )

    profit_margin = (
        total_profit / total_revenue
    ) * 100

    return {
        "total_revenue": round(total_revenue, 2),
        "total_profit": round(total_profit, 2),
        "total_orders": int(total_orders),
        "total_units": int(total_units),
        "average_order_value": round(
            average_order_value,
            2
        ),
        "profit_margin": round(
            profit_margin,
            2
        ),
    }


# ------------------------------------------------------------
# Revenue by Category
# ------------------------------------------------------------

def revenue_by_category(df):

    result = (
        df.groupby("Category")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
    )

    return result


# ------------------------------------------------------------
# Revenue by Region
# ------------------------------------------------------------

def revenue_by_region(df):

    result = (
        df.groupby("Region")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
    )

    return result


# ------------------------------------------------------------
# Monthly Sales Analysis
# ------------------------------------------------------------

def monthly_sales(df):

    result = (
        df.groupby(
            ["Year", "Month"]
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
    )

    result["Month_Date"] = pd.to_datetime(
        result["Year"].astype(str)
        + "-"
        + result["Month"].astype(str)
        + "-01"
    )

    result = result.sort_values(
        "Month_Date"
    )

    result["Month_Name"] = (
        result["Month_Date"]
        .dt.strftime("%b %Y")
    )

    return result


# ------------------------------------------------------------
# Quarterly Sales Analysis
# ------------------------------------------------------------

def quarterly_sales(df):

    result = (
        df.groupby(
            ["Year", "Quarter"]
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
    )

    return result


# ------------------------------------------------------------
# Top Products
# ------------------------------------------------------------

def top_products(df, limit=10):

    result = (
        df.groupby(
            [
                "Product_ID",
                "Product_Name",
                "Category",
            ]
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units=("Quantity", "sum"),
            Orders=("Order_ID", "nunique"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
        .head(limit)
    )

    return result


# ------------------------------------------------------------
# Bottom Products
# ------------------------------------------------------------

def bottom_products(df, limit=10):

    result = (
        df.groupby(
            [
                "Product_ID",
                "Product_Name",
                "Category",
            ]
        )
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units=("Quantity", "sum"),
            Orders=("Order_ID", "nunique"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=True
        )
        .head(limit)
    )

    return result


# ------------------------------------------------------------
# Customer Analysis
# ------------------------------------------------------------

def customer_analysis(df):

    result = (
        df.groupby("Customer_Type")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
    )

    result["Average_Order_Value"] = (
        result["Revenue"]
        / result["Orders"]
    )

    return result.sort_values(
        "Revenue",
        ascending=False
    )


# ------------------------------------------------------------
# Payment Method Analysis
# ------------------------------------------------------------

def payment_analysis(df):

    result = (
        df.groupby("Payment_Method")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("Order_ID", "nunique"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values(
            "Revenue",
            ascending=False
        )
    )

    return result


# ------------------------------------------------------------
# Discount Analysis
# ------------------------------------------------------------

def discount_analysis(df):

    result = (
        df.groupby("Discount")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique"),
        )
        .reset_index()
        .sort_values("Discount")
    )

    result["Profit_Margin"] = (
        result["Profit"]
        / result["Revenue"]
    ) * 100

    return result


# ------------------------------------------------------------
# Yearly Performance
# ------------------------------------------------------------

def yearly_performance(df):

    result = (
        df.groupby("Year")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order_ID", "nunique"),
            Units=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("Year")
    )

    result["Growth_Percentage"] = (
        result["Revenue"]
        .pct_change()
        * 100
    )

    result["Growth_Percentage"] = (
        result["Growth_Percentage"]
        .round(2)
    )

    return result


# ------------------------------------------------------------
# Main Analysis
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("RETAIL SALES BUSINESS INTELLIGENCE ANALYSIS")
    print("=" * 70)

    df = load_data()

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    kpis = calculate_kpis(df)

    print("\nKEY PERFORMANCE INDICATORS")
    print("-" * 70)

    print(
        f"Total Revenue      : ₹{kpis['total_revenue']:,.2f}"
    )

    print(
        f"Total Profit       : ₹{kpis['total_profit']:,.2f}"
    )

    print(
        f"Total Orders       : {kpis['total_orders']:,}"
    )

    print(
        f"Total Units Sold   : {kpis['total_units']:,}"
    )

    print(
        f"Average Order Value: ₹{kpis['average_order_value']:,.2f}"
    )

    print(
        f"Profit Margin      : {kpis['profit_margin']:.2f}%"
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    print("\nREVENUE BY CATEGORY")
    print("-" * 70)

    print(
        revenue_by_category(df).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Region
    # --------------------------------------------------------

    print("\nREVENUE BY REGION")
    print("-" * 70)

    print(
        revenue_by_region(df).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top Products
    # --------------------------------------------------------

    print("\nTOP 10 PRODUCTS")
    print("-" * 70)

    print(
        top_products(df).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Bottom Products
    # --------------------------------------------------------

    print("\nBOTTOM 10 PRODUCTS")
    print("-" * 70)

    print(
        bottom_products(df).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    print("\nCUSTOMER ANALYSIS")
    print("-" * 70)

    print(
        customer_analysis(df).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Payment
    # --------------------------------------------------------

    print("\nPAYMENT METHOD ANALYSIS")
    print("-" * 70)

    print(
        payment_analysis(df).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Yearly
    # --------------------------------------------------------

    print("\nYEARLY PERFORMANCE")
    print("-" * 70)

    print(
        yearly_performance(df).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("BUSINESS INTELLIGENCE ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()