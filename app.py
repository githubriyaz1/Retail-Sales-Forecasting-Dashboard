from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import forecasting
from src.business_insights import generate_insights


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_sales_forecasting_model.pkl"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .dashboard-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
    border-radius: 12px;
    padding: 18px;
    border: 1px solid rgba(128, 128, 128, 0.25);
}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dashboard_data():

    df = forecasting.load_data()

    return df


@st.cache_resource
def load_forecasting_model():

    model = joblib.load(
        MODEL_FILE
    )

    return model


# ============================================================
# FORMAT CURRENCY
# ============================================================

def format_currency(value):

    if value >= 1_000_000_000:

        return f"₹{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:

        return f"₹{value / 1_000_000:.2f}M"

    if value >= 1_000:

        return f"₹{value / 1_000:.2f}K"

    return f"₹{value:,.0f}"


# ============================================================
# LOAD
# ============================================================

df = load_dashboard_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Retail Sales Intelligence"
)

st.sidebar.caption(
    "Business Intelligence & Predictive Analytics"
)

st.sidebar.divider()

st.sidebar.markdown(
    "### 🎛️ Dashboard Filters"
)

st.sidebar.write(
    "Filter the historical sales analysis "
    "by region, category and customer type."
)


st.sidebar.divider()

st.sidebar.markdown(
    "### 📌 Quick Stats"
)

st.sidebar.metric(
    "Transactions",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Products",
    f"{df['Product_Name'].nunique():,}"
)

st.sidebar.metric(
    "Categories",
    f"{df['Category'].nunique():,}"
)

st.sidebar.markdown(
    "### Dashboard Filters"
)

st.sidebar.markdown(
    "Use the filters below to explore sales performance."
)


selected_regions = st.sidebar.multiselect(
    "Region",
    options=sorted(
        df["Region"].unique()
    ),
    default=sorted(
        df["Region"].unique()
    )
)


selected_categories = st.sidebar.multiselect(
    "Category",
    options=sorted(
        df["Category"].unique()
    ),
    default=sorted(
        df["Category"].unique()
    )
)


selected_customer_types = st.sidebar.multiselect(
    "Customer Type",
    options=sorted(
        df["Customer_Type"].unique()
    ),
    default=sorted(
        df["Customer_Type"].unique()
    )
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df["Region"].isin(
        selected_regions
    )
    &
    df["Category"].isin(
        selected_categories
    )
    &
    df["Customer_Type"].isin(
        selected_customer_types
    )
].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="dashboard-title">📊 Retail Sales Intelligence</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="dashboard-subtitle">'
    'Business Intelligence • Sales Analytics • Predictive Forecasting'
    '</p>',
    unsafe_allow_html=True
)

st.caption(
    "Interactive analytics platform for retail performance monitoring "
    "and six-month revenue forecasting."
)


# ============================================================
# TABS
# ============================================================

overview_tab, sales_tab, forecast_tab, insights_tab = st.tabs(
    [
        "📊 Executive Overview",
        "📈 Sales Analytics",
        "🔮 Sales Forecast",
        "💡 Business Insights"
    ]
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

with overview_tab:

    st.subheader(
        "Executive Overview"
    )

    if filtered_df.empty:

        st.warning(
            "No data available for the selected filters."
        )

    else:

        total_revenue = (
            filtered_df["Revenue"].sum()
        )

        total_profit = (
            filtered_df["Profit"].sum()
        )

        total_orders = (
            filtered_df["Order_ID"].nunique()
        )

        total_units = (
            filtered_df["Quantity"].sum()
        )

        average_order_value = (
            total_revenue
            / total_orders
        )

        profit_margin = (
            total_profit
            / total_revenue
        ) * 100


        
        # ============================================================
        # KPI CARDS
        # ============================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Revenue",
                format_currency(
                    total_revenue
                )
            )


        with col2:

            st.metric(
                "Total Profit",
                format_currency(
                    total_profit
                )
            )


        with col3:

            st.metric(
                "Orders",
                f"{total_orders:,}"
            )


        with col4:

            st.metric(
                "Units Sold",
                f"{total_units:,}"
            )


        margin_col, spacer = st.columns([1, 3])

        with margin_col:
            st.metric(
                "🎯 Profit Margin",
                f"{profit_margin:.2f}%"
            )




# ============================================================
# DATASET SUMMARY
# ============================================================

st.write("")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:

    st.info(
        f"📋 **Transactions**\n\n"
        f"{len(filtered_df):,}"
    )

with summary_col2:

    start_date = filtered_df["Order_Date"].min()

    end_date = filtered_df["Order_Date"].max()

    st.info(
        f"📅 **Data Period**\n\n"
        f"{start_date.strftime('%b %Y')} – "
        f"{end_date.strftime('%b %Y')}"
    )

with summary_col3:

    st.info(
        f"🏷️ **Categories**\n\n"
        f"{filtered_df['Category'].nunique()}"
    )

with summary_col4:

    st.info(
        f"🌎 **Regions**\n\n"
        f"{filtered_df['Region'].nunique()}"
    )




    st.divider()


        # MONTHLY TREND
    monthly = (
            filtered_df
            .groupby(
                filtered_df[
                    "Order_Date"
                ].dt.to_period("M")
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


    fig = px.line(
            monthly,
            x="Month",
            y="Revenue",
            markers=True,
            title="Monthly Revenue Trend"
        )


    fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue (₹)",
            hovermode="x unified"
        )


    st.plotly_chart(
            fig,
            use_container_width=True
        )


        # CATEGORY + REGION
    col1, col2 = st.columns(2)


    with col1:

            category_data = (
                filtered_df
                .groupby("Category")
                .agg(
                    Revenue=(
                        "Revenue",
                        "sum"
                    ),
                    Profit=(
                        "Profit",
                        "sum"
                    )
                )
                .reset_index()
                .sort_values(
                    "Revenue",
                    ascending=False
                )
            )


            fig_category = px.bar(
                category_data,
                x="Category",
                y="Revenue",
                title="Revenue by Category",
                text_auto=".2s"
            )


            fig_category.update_layout(
                xaxis_title="Category",
                yaxis_title="Revenue (₹)"
            )


            st.plotly_chart(
                fig_category,
                use_container_width=True
            )


    with col2:

            region_data = (
                filtered_df
                .groupby("Region")
                .agg(
                    Revenue=(
                        "Revenue",
                        "sum"
                    ),
                    Profit=(
                        "Profit",
                        "sum"
                    )
                )
                .reset_index()
                .sort_values(
                    "Revenue",
                    ascending=False
                )
            )


            fig_region = px.bar(
                region_data,
                x="Region",
                y="Revenue",
                title="Revenue by Region",
                text_auto=".2s"
            )


            fig_region.update_layout(
                xaxis_title="Region",
                yaxis_title="Revenue (₹)"
            )


            st.plotly_chart(
                fig_region,
                use_container_width=True
            )


# ============================================================
# SALES ANALYTICS
# ============================================================

with sales_tab:

    st.subheader(
        "Detailed Sales Analytics"
    )


    if filtered_df.empty:

        st.warning(
            "No data available for the selected filters."
        )

    else:

        # ----------------------------------------------------
        # PRODUCT PERFORMANCE
        # ----------------------------------------------------

        product_data = (
            filtered_df
            .groupby("Product_Name")
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                ),
                Profit=(
                    "Profit",
                    "sum"
                ),
                Units=(
                    "Quantity",
                    "sum"
                )
            )
            .reset_index()
            .sort_values(
                "Revenue",
                ascending=False
            )
        )


        st.subheader(
            "Top Products"
        )


        top_products = product_data.head(
            10
        )


        fig_products = px.bar(
            top_products,
            x="Revenue",
            y="Product_Name",
            orientation="h",
            title="Top 10 Products by Revenue",
            text_auto=".2s"
        )


        fig_products.update_layout(
            yaxis=dict(
                categoryorder="total ascending"
            )
        )


        st.plotly_chart(
            fig_products,
            use_container_width=True
        )


        # ----------------------------------------------------
        # CUSTOMER TYPE
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            customer_data = (
                filtered_df
                .groupby(
                    "Customer_Type"
                )["Revenue"]
                .sum()
                .reset_index()
            )


            fig_customer = px.pie(
                customer_data,
                names="Customer_Type",
                values="Revenue",
                title="Revenue by Customer Type"
            )


            st.plotly_chart(
                fig_customer,
                use_container_width=True
            )


        with col2:

            payment_data = (
                filtered_df
                .groupby(
                    "Payment_Method"
                )["Revenue"]
                .sum()
                .reset_index()
            )


            fig_payment = px.pie(
                payment_data,
                names="Payment_Method",
                values="Revenue",
                title="Revenue by Payment Method"
            )


            st.plotly_chart(
                fig_payment,
                use_container_width=True
            )


        # ----------------------------------------------------
        # REGION PROFITABILITY
        # ----------------------------------------------------

        regional_profit = (
            filtered_df
            .groupby("Region")
            .agg(
                Revenue=(
                    "Revenue",
                    "sum"
                ),
                Profit=(
                    "Profit",
                    "sum"
                )
            )
            .reset_index()
        )


        regional_profit[
            "Profit_Margin"
        ] = (
            regional_profit["Profit"]
            / regional_profit["Revenue"]
        ) * 100


        st.subheader(
            "Regional Profitability"
        )


        st.dataframe(
            regional_profit.style.format(
                {
                    "Revenue": "₹{:,.2f}",
                    "Profit": "₹{:,.2f}",
                    "Profit_Margin": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FORECAST
# ============================================================

with forecast_tab:

    st.subheader(
        "🔮 Revenue Forecast"
    )


    st.markdown(
    """
    ### 🔮 Future Revenue Prediction

    The forecasting engine predicts the expected revenue for the
    next six months using historical revenue patterns, seasonality,
    recent sales trends and lag-based features.
    """
    )

    st.info(
        "The prediction model uses historical monthly revenue, "
        "seasonality and lag-based features to forecast the "
        "next six months."
    )


    


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    metrics = forecasting.evaluate_model()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "MAPE",
            f"{metrics['MAPE']:.2f}%"
        )


    with col2:

        st.metric(
            "MAE",
            format_currency(
                metrics["MAE"]
            )
        )


    with col3:

        st.metric(
            "RMSE",
            format_currency(
                metrics["RMSE"]
            )
        )


    with col4:

        st.metric(
            "R² Score",
            f"{metrics['R2']:.4f}"
        )


    st.write("")

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.info(
        """
        **🤖 Forecasting Model**

        **Random Forest Regressor**

        The model was selected after comparing:

        - Linear Regression
        - Random Forest
        - Gradient Boosting

        Random Forest achieved the lowest MAPE of **7.30%**.
        """
    )


with model_col2:

    st.info(
        """
        **🧠 Forecasting Features**

        The model uses:

        - Time trend
        - Monthly seasonality
        - Previous-month revenue
        - Previous 2/3/6 month revenue
        - 3-month rolling average
        - 6-month rolling average
        """
    )

    st.divider()


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    full_monthly = (
        forecasting.prepare_monthly_data(
            df
        )
    )


    model = load_forecasting_model()


    future = forecasting.forecast_future(
        model,
        full_monthly,
        periods=6
    )


    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    fig_forecast = go.Figure()


    fig_forecast.add_trace(
        go.Scatter(
            x=full_monthly["Month"],
            y=full_monthly["Revenue"],
            mode="lines+markers",
            name="Historical Revenue"
        )
    )


    fig_forecast.add_trace(
        go.Scatter(
            x=future["Month"],
            y=future["Predicted_Revenue"],
            mode="lines+markers",
            name="Predicted Revenue",
            line=dict(
                dash="dash"
            )
        )
    )


    fig_forecast.update_layout(
        title="Historical Revenue vs 6-Month Forecast",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.subheader(
        "Next 6 Months Prediction"
    )


    forecast_table = future.copy()


    forecast_table["Month"] = (
        forecast_table["Month"]
        .dt.strftime("%B %Y")
    )


    forecast_table[
        "Predicted_Revenue"
    ] = (
        forecast_table[
            "Predicted_Revenue"
        ]
        .apply(
            lambda x:
            f"₹{x:,.2f}"
        )
    )


    forecast_table.columns = [
        "Month",
        "Predicted Revenue"
    ]


    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
# FORECAST SUMMARY
# --------------------------------------------------------

average_forecast = future[
    "Predicted_Revenue"
].mean()

highest_forecast = future.loc[
    future["Predicted_Revenue"].idxmax()
]

lowest_forecast = future.loc[
    future["Predicted_Revenue"].idxmin()
]


st.write("")

st.subheader(
    "Forecast Summary"
)


summary1, summary2, summary3 = st.columns(3)


with summary1:

    st.metric(
        "Average Monthly Forecast",
        format_currency(
            average_forecast
        )
    )


with summary2:

    st.metric(
        "Highest Forecast",
        format_currency(
            highest_forecast[
                "Predicted_Revenue"
            ]
        )
    )


with summary3:

    st.metric(
        "Lowest Forecast",
        format_currency(
            lowest_forecast[
                "Predicted_Revenue"
            ]
        )
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

with insights_tab:

    st.subheader(
        "💡 Automated Business Insights"
    )


    st.info(
        "These insights are automatically generated "
        "from the retail sales dataset."
    )


    insights = generate_insights(
        df
    )


    for index, insight in enumerate(
    insights,
    start=1
):

        with st.container(
        border=True
    ):

            st.markdown(
            f"**Insight {index}**"
        )

        st.write(
            insight
        )

    st.divider()


    st.subheader(
    "📚 Project Information"
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.markdown(
        """
        **Project Domain**

        Business Intelligence &
        Predictive Analytics
        """
    )

with info_col2:

    st.markdown(
        """
        **Analytics Type**

        Descriptive Analytics

        Predictive Analytics
        """
    )

with info_col3:

    st.markdown(
        """
        **Technology**

        Python • Pandas • Plotly
        • Scikit-learn • Streamlit
        """
    )

    st.subheader(
        "Management Recommendations"
    )


    recommendations = [
        (
            "📱 Electronics Focus",
            "Electronics contributes approximately "
            "76% of total revenue. Maintaining inventory "
            "availability and optimizing pricing in this "
            "category should be a priority."
        ),

        (
            "📍 Regional Growth",
            "The East region has the lowest revenue "
            "contribution. Targeted promotions and "
            "localized marketing could improve performance."
        ),

        (
            "💰 Discount Strategy",
            "The analysis indicates stronger profit margins "
            "at lower discount levels. Excessive discounting "
            "should therefore be avoided."
        ),

        (
            "📈 Forecast Planning",
            "The forecast indicates relatively stable monthly "
            "revenue over the next six months. Inventory and "
            "operational planning can use this forecast as "
            "a baseline."
        )
    ]


    for title, description in recommendations:

        with st.container(border=True):

            st.markdown(
                f"**{title}**"
            )

            st.write(
                description
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Retail Sales Intelligence | "
    "Business Intelligence & Predictive Analytics Project"
)