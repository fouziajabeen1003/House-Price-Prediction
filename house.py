import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HomeValue AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f8fc;
    }

    /* Hide Streamlit default menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.06);
        border: 1px solid #eef0f5;
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 18px;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #172033, #263b66);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(23,32,51,0.20);
    }

    .prediction-label {
        font-size: 15px;
        color: #cbd5e1;
        margin-bottom: 8px;
    }

    .prediction-price {
        font-size: 40px;
        font-weight: 800;
        color: white;
    }

    .prediction-sub {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-box {
        background: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #eef0f5;
        text-align: center;
    }

    .metric-title {
        font-size: 13px;
        color: #6b7280;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 700;
        color: #172033;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 48px;
        font-size: 16px;
        font-weight: 700;
        border: none;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #172033;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

@st.cache_data
def create_dataset():

    np.random.seed(42)

    n = 1000

    area = np.random.randint(500, 4500, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 5, n)
    age = np.random.randint(0, 30, n)
    parking = np.random.randint(0, 4, n)
    location_score = np.random.randint(1, 11, n)

    # Price generation
    price = (
        area * 4500
        + bedrooms * 700000
        + bathrooms * 500000
        + parking * 250000
        + location_score * 900000
        - age * 80000
        + np.random.normal(0, 800000, n)
    )

    price = np.maximum(price, 1000000)

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Parking": parking,
        "Location Score": location_score,
        "Price": price
    })

    return df


df = create_dataset()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return model, r2, mae


model, r2, mae = train_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>🏠</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align:center;'>HomeValue AI</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "",
        [
            "🏠 Price Prediction",
            "📊 Market Insights",
            "🤖 Model Performance"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; color:#cbd5e1;">
        <small>
        AI-powered house price estimation<br><br>
        Built with Python & Machine Learning
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">HomeValue AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the estimated market value of a property using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 - PRICE PREDICTION
# ============================================================

if page == "🏠 Price Prediction":

    col1, col2 = st.columns([1.35, 0.65], gap="large")

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    🏡 Property Information
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            area = st.number_input(
                "📐 Built-up Area (sq ft)",
                min_value=300,
                max_value=10000,
                value=1500,
                step=50
            )

            bedrooms = st.selectbox(
                "🛏️ Bedrooms",
                [1, 2, 3, 4, 5],
                index=2
            )

            bathrooms = st.selectbox(
                "🛁 Bathrooms",
                [1, 2, 3, 4],
                index=1
            )

        with c2:

            age = st.slider(
                "🏗️ Property Age (years)",
                min_value=0,
                max_value=30,
                value=5
            )

            parking = st.selectbox(
                "🚗 Parking Spaces",
                [0, 1, 2, 3],
                index=1
            )

            location_score = st.slider(
                "📍 Location Quality",
                min_value=1,
                max_value=10,
                value=7
            )

        st.markdown("<br>", unsafe_allow_html=True)

        predict_button = st.button(
            "✨ Estimate Property Value"
        )


    # --------------------------------------------------------
    # QUICK INFO
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    📋 Property Summary
                </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"**Built-up Area:** {area:,} sq ft")
        st.write(f"**Bedrooms:** {bedrooms}")
        st.write(f"**Bathrooms:** {bathrooms}")
        st.write(f"**Property Age:** {age} years")
        st.write(f"**Parking:** {parking}")
        st.write(f"**Location Score:** {location_score}/10")

        st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_data = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Age": [age],
            "Parking": [parking],
            "Location Score": [location_score]
        })

        prediction = model.predict(input_data)[0]

        price_per_sqft = prediction / area

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    ESTIMATED PROPERTY VALUE
                </div>

                <div class="prediction-price">
                    ₹ {prediction:,.0f}
                </div>

                <div class="prediction-sub">
                    Estimated using Random Forest Machine Learning
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Price / Sq Ft</div>
                    <div class="metric-value">
                        ₹{price_per_sqft:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Bedrooms</div>
                    <div class="metric-value">
                        {bedrooms}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Location Score</div>
                    <div class="metric-value">
                        {location_score}/10
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE 2 - MARKET INSIGHTS
# ============================================================

elif page == "📊 Market Insights":

    st.markdown(
        '<div class="card-title">📊 Market Insights</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="Area",
            y="Price",
            title="Property Area vs Price",
            labels={
                "Area": "Area (sq ft)",
                "Price": "Price (₹)"
            },
            opacity=0.6
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        avg_price = (
            df.groupby("Bedrooms")["Price"]
            .mean()
            .reset_index()
        )

        fig2 = px.bar(
            avg_price,
            x="Bedrooms",
            y="Price",
            title="Average Price by Bedrooms",
            labels={
                "Bedrooms": "Bedrooms",
                "Price": "Average Price (₹)"
            }
        )

        fig2.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# PAGE 3 - MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="card-title">🤖 Machine Learning Model</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Algorithm</div>
                <div class="metric-value">
                    Random Forest
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">R² Score</div>
                <div class="metric-value">
                    {r2:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Mean Absolute Error</div>
                <div class="metric-value">
                    ₹{mae:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">

        ### 🔍 How the Model Works

        The system uses a Random Forest Regression algorithm
        to estimate house prices based on property characteristics.

        **Input Features**

        • Built-up area  
        • Number of bedrooms  
        • Number of bathrooms  
        • Property age  
        • Parking spaces  
        • Location quality  

        **Machine Learning Pipeline**

        Property Details → Data Processing → Random Forest Model
        → Price Prediction

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#6b7280; padding:15px;">
        <small>
        HomeValue AI • House Price Prediction using Machine Learning
        </small>
    </div>
    """,
    unsafe_allow_html=True
)