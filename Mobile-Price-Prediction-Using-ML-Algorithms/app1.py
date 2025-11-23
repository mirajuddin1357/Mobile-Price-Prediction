# -------------------------------------------------
# MOBILE PRICE PREDICTION FULL STREAMLIT WEBSITE
# -------------------------------------------------

import streamlit as st
import pandas as pd
import sklearn
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image
import os
import base64

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Load the trained model
try:
    x = pickle.load(open(os.path.join(BASE_DIR, 'smartphone_price_model.pkl'), 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    x = None

# Load the dataset for encoding
try:
    dfen = pd.read_csv(os.path.join(BASE_DIR, 'smartphone_cleaned_v1.csv'))
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    x = None

# Encode categorical features
dfen = dfen[['brand_name', 'model', 'processor_brand', 'os']]
oe = OrdinalEncoder()
dfen['brand_name_enc'] = oe.fit_transform(dfen[['brand_name']])
dfen['model_enc'] = oe.fit_transform(dfen[['model']])
dfen['processor_brand_enc'] = oe.fit_transform(dfen[['processor_brand']])
dfen['os_enc'] = oe.fit_transform(dfen[['os']])

# Page Setup
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Mobile Price Prediction System")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Predict Price",
        "Documentation",
        'About Us'
    ]
)

# ===============================================================
# HOME PAGE
# ===============================================================
if page == "Home":
    try:
        with open(os.path.join(BASE_DIR, "image1.png"), "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()

        st.markdown(
            f"""
            <div style="
                text-align:center;
                background: linear-gradient(135deg, #000000, #1a1a1a);
                border:2px solid #ff0800;
                border-radius:18px;
                padding:25px;
                box-shadow:0 0 25px rgba(255,215,0,0.3);
                transition:all 0.4s ease;
            ">
            <h1 style='text-align:center; color:white; background:#111; padding:18px; border-radius:8px;'>
            Mobile Price Insights
            </h1>
                <img src="data:image/png;base64,{encoded_string}" width="680" 
                style="
                    border-radius:15px;
                    box-shadow:0 0 35px rgba(255,215,0,0.4);
                    transition: transform 0.4s ease, box-shadow 0.4s ease;
                " 
                onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 0 50px rgba(255, 215, 0, 1)';" 
                onmouseout="this.style.transform='scale(1.0)'; this.style.boxShadow='0 0 30px rgba(255,215,0,0.4)';">
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Hero image not found. Please ensure 'image1.png' is in the project directory.")

# ===============================================================
# PREDICTION PAGE
# ===============================================================
elif page == "Predict Price":
    st.title("Predict Mobile Price")

    # Load CSV for dropdowns
    df = pd.read_csv(os.path.join(BASE_DIR, 'smartphone_cleaned_v1.csv'))

    # Input Form
    brand = st.selectbox("Brand Name", df['brand_name'].unique())
    brand1 = dfen[dfen['brand_name']==brand]['brand_name_enc'].iloc[0]

    os_choice = st.selectbox('Operating System', df['os'].unique())
    os1 = dfen[dfen['os']==os_choice]['os_enc'].iloc[0]

    processor_brand = st.selectbox('Processor Brand', df[df['brand_name']== brand]['processor_brand'].unique())
    processor_brand1 = dfen[dfen['processor_brand']==processor_brand]['processor_brand_enc'].iloc[0]

    model_choice = st.selectbox('Model', df[df['brand_name'] == brand]['model'].unique())
    model1 = dfen[dfen['model']==model_choice]['model_enc'].iloc[0]

    ram = st.slider("RAM (GB)", min_value=2, max_value=16, step=2)
    cores = st.selectbox("Cores", [4, 6, 8])
    storage = st.selectbox("Storage (GB)", [4, 8, 16, 32, 64, 128, 256, 512])
    battery = st.selectbox("Battery (mAh)", df['battery_capacity'].unique())
    number_rear_camera = st.number_input('Number of Rear Cameras', min_value=1, max_value=5, step=1)
    number_front_camera = st.number_input('Number of Front Cameras', 1, 2)
    screen = st.number_input("Screen Size (Inches)", 4.5, 7.5, 6.4)
    rating = df[df['model'] == model_choice]['rating'].mean()
    fast_charge = df[df['model'] == model_choice]['fast_charging'].mean()
    refresh_rate = df[df['model'] == model_choice]['refresh_rate'].mean()
    primary_camera_rear = df[df['model'] == model_choice]['primary_camera_rear'].mean()
    extended_memory_available = df[df['model'] == model_choice]['extended_memory_available'].mean()
    extended_upto = df[df['model'] == model_choice]['extended_upto'].mean()
    resolution_width = df[df['model'] == model_choice]['resolution_width'].mean()
    resolution_height = df[df['model'] == model_choice]['resolution_height'].mean()
    processor_speed = st.number_input("Processor Speed (GHz)", 1.0, 3.5, 2.2)

    col1, col2, col3 = st.columns(3)
    fiveG = col1.checkbox("5G Support")
    fiveG = 1 if fiveG else 0
    nfc = col2.checkbox("NFC")
    nfc = 1 if nfc else 0
    ir = col3.checkbox("IR Blaster")
    ir = 1 if ir else 0

    if st.button("Predict Price"):
        price = x.predict([[brand1, model1, rating, fiveG, nfc, ir,
                            processor_brand1, cores, processor_speed, battery,
                            fast_charge, fast_charge, ram,
                            storage, screen, int(refresh_rate), number_rear_camera,
                            number_front_camera, os1, primary_camera_rear,
                            primary_camera_rear, int(extended_memory_available), extended_upto,
                            int(resolution_width), int(resolution_height)]])
        st.success(f"Predicted Price: **PKR {int(price*3.17):,}**")

# ===============================================================
# DOCUMENTATION PAGE
# ===============================================================
elif page == "Documentation":
    st.title("Project Documentation")
    with open(os.path.join(BASE_DIR, "image.png"), "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <div style="
                text-align:center;
                background: linear-gradient(135deg, #000000, #1a1a1a);
                border:2px solid #ff0800;
                border-radius:18px;
                padding:25px;
                box-shadow:0 0 25px rgba(255,215,0,0.3);
                transition:all 0.4s ease;
            ">
            <h1 style='text-align:center; color:white; background:#111; padding:18px; border-radius:8px;'>
            Mobile Price Insights
            </h1>
                <img src="data:image/png;base64,{encoded_string}" width="680">
            </div>
            """,
            unsafe_allow_html=True
        )

# ===============================================================
# CONTACT PAGE
# ===============================================================
elif page == "About Us":
    st.markdown(
        """
        <div style="border-radius:12px; border:2px solid red; padding:20px; background:linear-gradient(135deg, #1a1a1a, #000000); color:white; font-family:'Poppins', sans-serif;">
        <h2 style="text-align:center; color:#FF4500;">Meet the AI Engineers</h2>
        <h3 style="color:#FF6347;">Musa Khan</h3>
        <p>My name is Musa Khan, and I am a professional AI Engineer specializing in machine learning, deep learning, and data-driven solutions. I work with Python and modern AI frameworks to build intelligent systems that solve real-world problems.</p>
        <h3 style="color:#FF6347;">Miraj Ud Din</h3>
        <p>Miraj Ud Din is a BS (AI) student from Peshawar, specializing in Data Science, Machine Learning, Deep Learning and AI. He focuses on developing intelligent systems and deploying ML models effectively.</p>
        <h3 style="color:#FF6347;">Ahmad Aziz</h3>
        <p>Ahmad Aziz is an experienced AI engineer specializing in deep learning and data-driven solutions. He excels in building and deploying ML applications with high performance.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
