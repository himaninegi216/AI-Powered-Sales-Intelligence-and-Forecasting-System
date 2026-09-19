import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# ================= SIDEBAR =================

st.sidebar.title("🤖 AI Sales Intelligence")

st.sidebar.write("Navigate through the dashboard")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 Sales Analysis",
        "👥 Customer Intelligence",
        "🔮 Forecasting",
        "💬 Sentiment Analysis",
        "🤖 AI Insights"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI-Powered Sales Intelligence System")

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Sales Intelligence",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD SALES DATA
# -----------------------------
df = pd.read_csv("DATA/train.csv")
# Load trained forecasting model
forecast_model = joblib.load("MODELS/sales_forecasting_model.pkl")

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# -----------------------------
# CALCULATE KPIs
# -----------------------------
total_sales = df["Sales"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()
average_order_value = total_sales / total_orders

# -----------------------------
# DASHBOARD HEADER
# -----------------------------
# ================= PAGE NAVIGATION =================

if page == "🏠 Overview":
    st.title("🤖 AI-Powered Sales Intelligence System")

elif page == "📊 Sales Analysis":
    st.title("📊 Sales Analysis")

elif page == "👥 Customer Intelligence":
    st.title("👥 Customer Intelligence")

elif page == "🔮 Forecasting":
    st.title("🔮 Sales Forecasting")

elif page == "💬 Sentiment Analysis":
    st.title("💬 Customer Sentiment Analysis")

elif page == "🤖 AI Insights":
    st.title("🤖 AI Business Insights")

st.write(
    "An intelligent dashboard for sales analysis, customer segmentation, "
    "forecasting, sentiment analysis, and AI-powered business insights."
)

st.divider()

# -----------------------------
# KPI CARDS
# -----------------------------
# ================= CUSTOM DASHBOARD STYLE =================

st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    text-align: center;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 15px;
    color: #6b7280;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# KPI CARDS
# -----------------------------

if page == "🏠 Overview":

    st.subheader("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Total Sales</div>
            <div class="metric-value">₹{total_sales:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🛒 Total Orders</div>
            <div class="metric-value">{total_orders:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">👥 Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📦 Average Order Value</div>
            <div class="metric-value">₹{average_order_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.success("✅ Sales data loaded successfully!")
# -----------------------------
# -----------------------------
# SALES ANALYSIS
# -----------------------------

if page == "📊 Sales Analysis":

    st.divider()

    # ================= SALES OVERVIEW CHARTS =================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Monthly Sales Trend")

        monthly_sales = (
            df.set_index("Order Date")
            .resample("ME")["Sales"]
            .sum()
        )

        st.line_chart(monthly_sales)

    with col2:
        st.subheader("📊 Sales by Category")

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(category_sales)

    # ================= TOP 10 PRODUCTS =================

    st.subheader("🏆 Top 10 Products")

    top_products = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_products)


# -----------------------------
# SALES FORECASTING
# -----------------------------

if page == "🔮 Forecasting":

    st.divider()

    st.subheader("🔮 6-Month Sales Forecast")

    # Create monthly sales data

    monthly_sales = (
        df.set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
    )

    # Prepare forecasting data

    forecast_data = monthly_sales.reset_index()

    forecast_data.columns = ["Date", "Sales"]

    forecast_data["Year"] = forecast_data["Date"].dt.year
    forecast_data["Month"] = forecast_data["Date"].dt.month
    forecast_data["Month_Number"] = range(1, len(forecast_data) + 1)

    # Create lag features

    forecast_data["Sales_Lag1"] = forecast_data["Sales"].shift(1)
    forecast_data["Sales_Lag2"] = forecast_data["Sales"].shift(2)
    forecast_data["Rolling_Mean_3"] = forecast_data["Sales"].rolling(3).mean()

    # Remove missing rows

    forecast_data = forecast_data.dropna().reset_index(drop=True)

    # Get latest values for forecasting

    last_row = forecast_data.iloc[-1]

    last_lag1 = last_row["Sales"]
    last_lag2 = forecast_data.iloc[-2]["Sales"]

    last_rolling = forecast_data.tail(3)["Sales"].mean()

    last_date = monthly_sales.index[-1]

    future_predictions = []

    # Generate next 6 months

    for i in range(1, 7):

        future_date = last_date + pd.DateOffset(months=i)

        future_year = future_date.year
        future_month = future_date.month

        future_month_number = len(monthly_sales) + i

        input_data = pd.DataFrame({
            "Year": [future_year],
            "Month": [future_month],
            "Month_Number": [future_month_number],
            "Sales_Lag1": [last_lag1],
            "Sales_Lag2": [last_lag2],
            "Rolling_Mean_3": [last_rolling]
        })

        prediction = forecast_model.predict(input_data)[0]

        future_predictions.append({
            "Date": future_date,
            "Predicted_Sales": prediction
        })

        # Update lag values

        last_lag2 = last_lag1
        last_lag1 = prediction

        last_rolling = (
            last_lag1 + last_lag2 + last_rolling
        ) / 3

    # Create forecast DataFrame

    future_df = pd.DataFrame(future_predictions)

    # Display forecast table

    st.dataframe(
        future_df.style.format({
            "Predicted_Sales": "₹{:,.2f}"
        }),
        use_container_width=True
    )

    # Forecast chart

    st.line_chart(
        future_df.set_index("Date")["Predicted_Sales"]
    )


# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

if page == "👥 Customer Intelligence":
    st.divider()
    st.subheader("👥 Customer Segmentation")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Create RFM data
    reference_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "Order Date": lambda x: (reference_date - x.max()).days,
        "Order ID": "nunique",
        "Sales": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Handle missing values
    rfm["Recency"] = rfm["Recency"].fillna(rfm["Recency"].median())

    # Scale RFM values
    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    # Apply K-Means clustering
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Name the customer segments
    cluster_names = {
        0: "Regular / Low-Value",
        1: "High-Value",
        2: "At-Risk",
        3: "Loyal / Frequent"
    }

    rfm["Customer_Segment"] = rfm["Cluster"].map(cluster_names)

    # Segment counts
    segment_counts = (
        rfm["Customer_Segment"]
        .value_counts()
        .rename_axis("Customer_Segment")
        .reset_index(name="Customers")
    )

    # Segment sales
    segment_sales = (
        rfm.groupby("Customer_Segment")["Monetary"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💎 High-Value",
        segment_counts.loc[
            segment_counts["Customer_Segment"] == "High-Value",
            "Customers"
        ].iloc[0]
    )

    col2.metric(
        "👑 Loyal / Frequent",
        segment_counts.loc[
            segment_counts["Customer_Segment"] == "Loyal / Frequent",
            "Customers"
        ].iloc[0]
    )

    col3.metric(
        "⚠️ At-Risk",
        segment_counts.loc[
            segment_counts["Customer_Segment"] == "At-Risk",
            "Customers"
        ].iloc[0]
    )

    col4.metric(
        "👤 Regular / Low-Value",
        segment_counts.loc[
            segment_counts["Customer_Segment"] == "Regular / Low-Value",
            "Customers"
        ].iloc[0]
    )

    # Customer count chart
    st.write("### Customer Distribution by Segment")

    st.bar_chart(
        segment_counts.set_index("Customer_Segment")["Customers"]
    )

    # Sales contribution
    st.write("### Sales Contribution by Segment")

    st.bar_chart(
        segment_sales.set_index("Customer_Segment")["Monetary"]
    )
# -----------------------------
# -----------------------------
# SENTIMENT ANALYSIS
# -----------------------------

if page == "💬 Sentiment Analysis":

    st.divider()
    st.subheader("💬 Customer Sentiment Analysis")

    # Load customer review dataset
    reviews_df = pd.read_csv("DATA/Dataset-SA.csv")

    # Convert rating to numeric
    reviews_df["Rate"] = pd.to_numeric(
        reviews_df["Rate"],
        errors="coerce"
    )

    # Sentiment distribution
    sentiment_counts = (
        reviews_df["Sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    # Average rating by sentiment
    sentiment_rating = (
        reviews_df.groupby("Sentiment")["Rate"]
        .mean()
        .round(2)
        .reset_index(name="Average_Rating")
    )

    # Display sentiment metrics
    positive_count = sentiment_counts.loc[
        sentiment_counts["Sentiment"].str.lower() == "positive",
        "Reviews"
    ].sum()

    negative_count = sentiment_counts.loc[
        sentiment_counts["Sentiment"].str.lower() == "negative",
        "Reviews"
    ].sum()

    neutral_count = sentiment_counts.loc[
        sentiment_counts["Sentiment"].str.lower() == "neutral",
        "Reviews"
    ].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("😊 Positive Reviews", f"{positive_count:,}")
    col2.metric("😡 Negative Reviews", f"{negative_count:,}")
    col3.metric("😐 Neutral Reviews", f"{neutral_count:,}")

    # Sentiment distribution chart
    st.write("### Sentiment Distribution")

    st.bar_chart(
        sentiment_counts.set_index("Sentiment")["Reviews"]
    )

    # Average rating chart
    st.write("### Average Rating by Sentiment")

    st.bar_chart(
        sentiment_rating.set_index("Sentiment")["Average_Rating"]
    )

# ================= AI BUSINESS INSIGHTS =================

if page == "🤖 AI Insights":

    st.divider()
    st.subheader("🤖 AI Business Insights")

    # Customer segmentation data
    reference_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm_ai = df.groupby("Customer ID").agg({
        "Order Date": lambda x: (reference_date - x.max()).days,
        "Order ID": "nunique",
        "Sales": "sum"
    })

    rfm_ai.columns = ["Recency", "Frequency", "Monetary"]
    rfm_ai["Recency"] = rfm_ai["Recency"].fillna(
        rfm_ai["Recency"].median()
    )

    scaler_ai = StandardScaler()

    rfm_scaled_ai = scaler_ai.fit_transform(
        rfm_ai[["Recency", "Frequency", "Monetary"]]
    )

    kmeans_ai = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm_ai["Cluster"] = kmeans_ai.fit_predict(rfm_scaled_ai)

    cluster_names_ai = {
        0: "Regular / Low-Value",
        1: "High-Value",
        2: "At-Risk",
        3: "Loyal / Frequent"
    }

    rfm_ai["Customer_Segment"] = rfm_ai["Cluster"].map(
        cluster_names_ai
    )

    segment_counts_ai = (
        rfm_ai["Customer_Segment"]
        .value_counts()
        .rename_axis("Customer_Segment")
        .reset_index(name="Customers")
    )

    segment_sales_ai = (
        rfm_ai.groupby("Customer_Segment")["Monetary"]
        .sum()
        .reset_index()
    )

    at_risk = segment_counts_ai.loc[
        segment_counts_ai["Customer_Segment"] == "At-Risk",
        "Customers"
    ].iloc[0]

    high_value = segment_counts_ai.loc[
        segment_counts_ai["Customer_Segment"] == "High-Value",
        "Customers"
    ].iloc[0]

    loyal_sales = segment_sales_ai.loc[
        segment_sales_ai["Customer_Segment"] == "Loyal / Frequent",
        "Monetary"
    ].iloc[0]

    # Sentiment data
    reviews_ai = pd.read_csv("DATA/Dataset-SA.csv")

    reviews_ai["Rate"] = pd.to_numeric(
        reviews_ai["Rate"],
        errors="coerce"
    )

    sentiment_rating_ai = (
        reviews_ai.groupby("Sentiment")["Rate"]
        .mean()
        .round(2)
        .reset_index(name="Average_Rating")
    )

    negative_rating = sentiment_rating_ai.loc[
        sentiment_rating_ai["Sentiment"].str.lower() == "negative",
        "Average_Rating"
    ].iloc[0]

    # Next month's sales prediction
    monthly_sales_ai = (
        df.set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
    )

    forecast_data_ai = monthly_sales_ai.reset_index()
    forecast_data_ai.columns = ["Date", "Sales"]

    forecast_data_ai["Year"] = forecast_data_ai["Date"].dt.year
    forecast_data_ai["Month"] = forecast_data_ai["Date"].dt.month
    forecast_data_ai["Month_Number"] = range(
        1, len(forecast_data_ai) + 1
    )

    forecast_data_ai["Sales_Lag1"] = (
        forecast_data_ai["Sales"].shift(1)
    )

    forecast_data_ai["Sales_Lag2"] = (
        forecast_data_ai["Sales"].shift(2)
    )

    forecast_data_ai["Rolling_Mean_3"] = (
        forecast_data_ai["Sales"].rolling(3).mean()
    )

    forecast_data_ai = forecast_data_ai.dropna()

    last_row_ai = forecast_data_ai.iloc[-1]

    last_date_ai = monthly_sales_ai.index[-1]

    next_date_ai = last_date_ai + pd.DateOffset(months=1)

    input_ai = pd.DataFrame({
        "Year": [next_date_ai.year],
        "Month": [next_date_ai.month],
        "Month_Number": [len(monthly_sales_ai) + 1],
        "Sales_Lag1": [last_row_ai["Sales"]],
        "Sales_Lag2": [forecast_data_ai.iloc[-2]["Sales"]],
        "Rolling_Mean_3": [
            forecast_data_ai.tail(3)["Sales"].mean()
        ]
    })

    next_month_sales = forecast_model.predict(input_ai)[0]

    # AI insights
    st.warning(
        f"⚠️ {at_risk} customers are currently At-Risk. "
        "Consider targeted retention campaigns and personalized offers."
    )

    st.info(
        f"💎 {high_value} High-Value customers identified. "
        "Consider loyalty benefits and premium offers."
    )

    st.success(
        f"❤️ Loyal / Frequent customers contributed "
        f"₹{loyal_sales:,.2f} in sales."
    )

    st.error(
        f"⚠️ Negative reviews have an average rating of "
        f"{negative_rating:.2f}. "
        "Customer complaints should be investigated."
    )

    st.info(
        f"🔮 Next month's predicted sales: "
        f"₹{next_month_sales:,.2f}"
    )