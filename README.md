# Retail Sales Intelligence

## Business Intelligence & Predictive Analytics Dashboard

A Business Intelligence and Predictive Analytics project developed to analyze retail sales performance, identify important business trends, and forecast future revenue using machine learning.

The project provides an interactive Streamlit dashboard containing historical sales analysis, business KPIs, interactive visualizations, automated business insights, and six-month revenue forecasting.

---

## 1. Project Overview

Retail businesses generate large amounts of transactional data. Analyzing this data manually can make it difficult to identify sales trends, profitable product categories, strong and weak regions, customer behavior, and future revenue expectations.

This project solves this problem by developing an interactive Business Intelligence platform that transforms raw retail transaction data into meaningful business information.

The system performs:

- Data cleaning
- Exploratory Data Analysis
- Business Intelligence analysis
- KPI calculation
- Sales trend analysis
- Product category analysis
- Regional analysis
- Customer analysis
- Business insight generation
- Machine learning model comparison
- Revenue forecasting
- Interactive dashboard visualization

---

## 2. Project Title

**Retail Sales Intelligence: Business Intelligence and Predictive Revenue Analytics**

---

## 3. Objectives

The main objectives of this project are:

1. To analyze retail sales transaction data.
2. To identify important sales and profitability trends.
3. To compare performance across product categories.
4. To analyze regional sales performance.
5. To understand customer purchasing behavior.
6. To calculate important business KPIs.
7. To generate automated business insights.
8. To compare multiple machine learning models.
9. To forecast future monthly revenue.
10. To develop an interactive Business Intelligence dashboard.

---

## 4. Dataset

The project uses a synthetic retail sales dataset containing 10,000 transaction records.

### Dataset Attributes

| Attribute | Description |
|---|---|
| Order_ID | Unique order identifier |
| Order_Date | Date of transaction |
| Product_ID | Product identifier |
| Product_Name | Product name |
| Category | Product category |
| Region | Sales region |
| Customer_Type | Customer classification |
| Quantity | Number of units sold |
| Unit_Price | Price per unit |
| Discount | Discount percentage |
| Revenue | Revenue generated |
| Cost | Product cost |
| Profit | Profit generated |
| Payment_Method | Payment method |

### Dataset Period

**January 2024 – September 2026**

### Records

**10,000 transactions**

---

## 5. Technology Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Plotly

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Dashboard

- Streamlit

### Development Environment

- Visual Studio Code
- Jupyter Notebook
- Python Virtual Environment

---

## 6. System Architecture

The project follows the following data processing pipeline:

Raw Retail Data

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Business Intelligence Analysis

↓

Feature Engineering

↓

Machine Learning Model Comparison

↓

Best Model Selection

↓

Revenue Forecasting

↓

Streamlit Dashboard

↓

Business Insights & Recommendations

---

## 7. Data Processing

The raw dataset is processed before analysis.

The data processing stage includes:

- Loading the dataset
- Checking missing values
- Converting date columns
- Validating numerical columns
- Removing invalid records
- Preparing the cleaned dataset
- Creating aggregated datasets for analysis

The cleaned dataset is stored in:

```text
data/processed/retail_sales_clean.csv