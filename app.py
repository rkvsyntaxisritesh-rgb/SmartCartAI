import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartCart AI", page_icon="🛒")

@st.cache_resource
def load_model():
    return joblib.load("smartcart_sales_model.joblib")

model = load_model()

st.title("SmartCart AI — Sales Predictor")
st.write("Enter product details to estimate units sold.")

category = st.selectbox(
    "Category",
    [
        "Beauty & Personal Care",
        "Books",
        "Electronics",
        "Fashion",
        "Grocery",
        "Home & Kitchen",
        "Sports & Fitness",
        "Toys & Games",
    ],
)
product_type = st.selectbox("Product type", ["Budget", "Mid-range", "Premium"])
platform = st.selectbox("Platform", ["Amazon", "Flipkart", "Meesho", "Myntra", "Nykaa"])
price = st.number_input("Price", min_value=0.0, value=1000.0)
discount_percent = st.slider("Discount (%)", 0, 60, 15)
rating = st.slider("Rating", 2.8, 5.0, 4.3, step=0.1)
delivery_days = st.slider("Delivery days", 1, 8, 3)

if st.button("Estimate sales"):
    product = pd.DataFrame([{
        "category": category,
        "type": product_type,
        "platform": platform,
        "price": price,
        "discount_percent": discount_percent,
        "rating": rating,
        "delivery_days": delivery_days,
    }])
    prediction = model.predict(product)[0]
    st.success(f"Estimated units sold: {prediction:.0f}")