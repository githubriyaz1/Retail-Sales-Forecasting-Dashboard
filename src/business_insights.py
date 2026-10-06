from pathlib import Path

import pandas as pd


# ============================================================
# RETAIL SALES BUSINESS INSIGHTS ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_sales_clean.csv"
)


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
# Generate Business Insights
# ------------------------------------------------------------

def generate_insights(df):

    insights = []

    # ========================================================
    # Overall Performance
    # ========================================================

    total_revenue = df["Revenue"].sum()

    total_profit = df["Profit"].sum()

    profit_margin = (
        total_profit
        / total_revenue
    ) * 100

    insights.append(
        f"The business generated total revenue of "
        f"₹{total_revenue:,.2f} with a profit of "
        f"₹{total_profit:,.2f}, resulting in an overall "
        f"profit margin of {profit_margin:.2f}%."
    )

    # ========================================================
    # Category Insight
    # ========================================================

    category = (
        df.groupby("Category")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_category = category.index[0]

    top_category_revenue = (
        category.iloc[0]
    )

    category_share = (
        top_category_revenue
        / total_revenue
    ) * 100

    insights.append(
        f"{top_category} is the highest-performing "
        f"product category, generating "
        f"₹{top_category_revenue:,.2f}, which represents "
        f"{category_share:.2f}% of total revenue."
    )

    # ========================================================
    # Region Insight
    # ========================================================

    region = (
        df.groupby("Region")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_region = region.index[0]

    top_region_revenue = (
        region.iloc[0]
    )

    region_share = (
        top_region_revenue
        / total_revenue
    ) * 100

    insights.append(
        f"The {top_region} region is the strongest "
        f"market, contributing "
        f"₹{top_region_revenue:,.2f}, or "
        f"{region_share:.2f}% of total revenue."
    )

    # ========================================================
    # Weakest Region
    # ========================================================

    weakest_region = region.index[-1]

    weakest_region_revenue = (
        region.iloc[-1]
    )

    insights.append(
        f"The {weakest_region} region has the lowest "
        f"revenue contribution at "
        f"₹{weakest_region_revenue:,.2f}, indicating "
        f"an opportunity for targeted regional growth."
    )

    # ========================================================
    # Top Product
    # ========================================================

    products = (
        df.groupby("Product_Name")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_product = products.index[0]

    top_product_revenue = (
        products.iloc[0]
    )

    insights.append(
        f"The highest-revenue product is "
        f"{top_product}, generating "
        f"₹{top_product_revenue:,.2f} in revenue."
    )

    # ========================================================
    # Customer Type
    # ========================================================

    customer = (
        df.groupby("Customer_Type")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_customer_type = customer.index[0]

    top_customer_revenue = (
        customer.iloc[0]
    )

    customer_share = (
        top_customer_revenue
        / total_revenue
    ) * 100

    insights.append(
        f"{top_customer_type} customers contribute "
        f"the highest revenue at "
        f"₹{top_customer_revenue:,.2f}, accounting for "
        f"{customer_share:.2f}% of total revenue."
    )

    # ========================================================
    # Payment Method
    # ========================================================

    payment = (
        df.groupby("Payment_Method")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_payment = payment.index[0]

    top_payment_revenue = (
        payment.iloc[0]
    )

    insights.append(
        f"{top_payment} is the leading payment method "
        f"by revenue, contributing "
        f"₹{top_payment_revenue:,.2f}."
    )

    # ========================================================
    # Discount Insight
    # ========================================================

    discount = (
        df.groupby("Discount")
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    discount["Profit_Margin"] = (
        discount["Profit"]
        / discount["Revenue"]
    ) * 100

    best_margin_row = (
        discount.loc[
            discount["Profit_Margin"].idxmax()
        ]
    )

    insights.append(
        f"The highest average profit margin occurs "
        f"at a {best_margin_row['Discount']:.0f}% "
        f"discount level, with a margin of "
        f"{best_margin_row['Profit_Margin']:.2f}%."
    )

    # ========================================================
    # Monthly Performance
    # ========================================================

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

    best_month = (
        monthly.loc[
            monthly["Revenue"].idxmax()
        ]
    )

    weakest_month = (
        monthly.loc[
            monthly["Revenue"].idxmin()
        ]
    )

    insights.append(
        f"The strongest sales month in the historical "
        f"dataset was {best_month['Month']}, generating "
        f"₹{best_month['Revenue']:,.2f}."
    )

    insights.append(
        f"The weakest sales month was "
        f"{weakest_month['Month']}, generating "
        f"₹{weakest_month['Revenue']:,.2f}."
    )

    return insights


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 75)
    print("RETAIL SALES BUSINESS INSIGHTS")
    print("=" * 75)

    df = load_data()

    insights = generate_insights(
        df
    )

    print()

    for number, insight in enumerate(
        insights,
        start=1
    ):

        print(
            f"{number}. {insight}"
        )

    print("\n" + "=" * 75)
    print("BUSINESS INSIGHTS GENERATED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()