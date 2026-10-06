import os
import random
import numpy as np
import pandas as pd


# ============================================================
# RETAIL SALES DATASET GENERATOR
# ============================================================

np.random.seed(42)
random.seed(42)

NUMBER_OF_RECORDS = 10000


# ------------------------------------------------------------
# Product Master Data
# ------------------------------------------------------------

products = [
    ("P001", "Laptop", "Electronics", 55000, 90000),
    ("P002", "Smartphone", "Electronics", 18000, 35000),
    ("P003", "Tablet", "Electronics", 15000, 30000),
    ("P004", "Monitor", "Electronics", 10000, 22000),
    ("P005", "Keyboard", "Electronics", 1200, 3500),
    ("P006", "Mouse", "Electronics", 700, 2500),
    ("P007", "Headphones", "Electronics", 1500, 6000),
    ("P008", "Smart Watch", "Electronics", 3000, 12000),

    ("P009", "T-Shirt", "Clothing", 500, 1500),
    ("P010", "Jeans", "Clothing", 1000, 3000),
    ("P011", "Jacket", "Clothing", 1800, 5000),
    ("P012", "Shoes", "Clothing", 1500, 6000),
    ("P013", "Hoodie", "Clothing", 1000, 3000),
    ("P014", "Formal Shirt", "Clothing", 800, 2500),

    ("P015", "Office Chair", "Furniture", 5000, 15000),
    ("P016", "Study Table", "Furniture", 6000, 18000),
    ("P017", "Bookshelf", "Furniture", 4000, 12000),
    ("P018", "Sofa", "Furniture", 20000, 60000),
    ("P019", "Bed", "Furniture", 15000, 50000),

    ("P020", "Rice", "Grocery", 800, 1500),
    ("P021", "Cooking Oil", "Grocery", 1000, 1800),
    ("P022", "Coffee", "Grocery", 300, 900),
    ("P023", "Tea", "Grocery", 250, 800),
    ("P024", "Snacks", "Grocery", 100, 500),
    ("P025", "Beverages", "Grocery", 100, 600),

    ("P026", "Backpack", "Accessories", 800, 2500),
    ("P027", "Wallet", "Accessories", 500, 2000),
    ("P028", "Belt", "Accessories", 400, 1500),
    ("P029", "Sunglasses", "Accessories", 700, 3000),
    ("P030", "Travel Bag", "Accessories", 1500, 5000),
]


# ------------------------------------------------------------
# Other Business Data
# ------------------------------------------------------------

regions = [
    "South",
    "North",
    "West",
    "East",
]

customer_types = [
    "Regular",
    "Premium",
    "New",
]

payment_methods = [
    "UPI",
    "Credit Card",
    "Debit Card",
    "Cash",
    "Net Banking",
]


# ------------------------------------------------------------
# Create Product DataFrame
# ------------------------------------------------------------

product_df = pd.DataFrame(
    products,
    columns=[
        "Product_ID",
        "Product_Name",
        "Category",
        "Min_Price",
        "Max_Price",
    ],
)


# ------------------------------------------------------------
# Generate Sales Records
# ------------------------------------------------------------

records = []

date_range = pd.date_range(
    start="2024-01-01",
    end="2026-09-30",
    freq="D",
)


for i in range(NUMBER_OF_RECORDS):

    product = product_df.sample(
        n=1,
        weights=[
            12, 11, 9, 8, 7, 6, 7, 6,
            8, 7, 5, 7, 5, 5,
            5, 4, 3, 2, 3,
            7, 6, 5, 5, 9, 8,
            7, 5, 5, 4, 3,
        ],
    ).iloc[0]

    order_date = random.choice(date_range)

    category = product["Category"]

    # Quantity depends on category
    if category == "Grocery":
        quantity = np.random.randint(1, 15)
    elif category == "Clothing":
        quantity = np.random.randint(1, 8)
    elif category == "Accessories":
        quantity = np.random.randint(1, 7)
    elif category == "Electronics":
        quantity = np.random.randint(1, 5)
    else:
        quantity = np.random.randint(1, 4)

    unit_price = np.random.uniform(
        product["Min_Price"],
        product["Max_Price"],
    )

    unit_price = round(unit_price, 2)

    discount = random.choices(
        [0, 5, 10, 15, 20],
        weights=[30, 30, 20, 15, 5],
    )[0]

    region = random.choices(
        regions,
        weights=[35, 25, 25, 15],
    )[0]

    customer_type = random.choices(
        customer_types,
        weights=[60, 20, 20],
    )[0]

    payment_method = random.choice(payment_methods)

    gross_amount = quantity * unit_price

    discount_amount = gross_amount * (discount / 100)

    revenue = gross_amount - discount_amount

    # Cost is approximately 65–88% of revenue
    cost_percentage = np.random.uniform(0.65, 0.88)

    cost = revenue * cost_percentage

    profit = revenue - cost

    records.append(
        {
            "Order_ID": f"ORD{i + 1:05d}",
            "Order_Date": order_date,
            "Product_ID": product["Product_ID"],
            "Product_Name": product["Product_Name"],
            "Category": category,
            "Region": region,
            "Customer_Type": customer_type,
            "Quantity": quantity,
            "Unit_Price": round(unit_price, 2),
            "Discount": discount,
            "Revenue": round(revenue, 2),
            "Cost": round(cost, 2),
            "Profit": round(profit, 2),
            "Payment_Method": payment_method,
        }
    )


# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(records)


# ------------------------------------------------------------
# Sort by Date
# ------------------------------------------------------------

df = df.sort_values("Order_Date").reset_index(drop=True)


# ------------------------------------------------------------
# Save Dataset
# ------------------------------------------------------------

output_directory = "data/raw"

os.makedirs(output_directory, exist_ok=True)

output_file = os.path.join(
    output_directory,
    "retail_sales_raw.csv",
)

df.to_csv(
    output_file,
    index=False,
)


# ------------------------------------------------------------
# Display Dataset Information
# ------------------------------------------------------------

print("=" * 60)
print("RETAIL SALES DATASET GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Date Range : {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print(f"\nDataset saved to:")
print(output_file)

print("=" * 60)