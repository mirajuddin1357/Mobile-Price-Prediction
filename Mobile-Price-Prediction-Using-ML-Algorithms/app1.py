import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# Base directory (relative path)
BASE_DIR = os.path.dirname(__file__)

# Load dataset
csv_path = os.path.join(BASE_DIR, "smartphone_cleaned_v1.csv")
df = pd.read_csv(csv_path)

# Display dataframe in app
st.title("📱 Mobile Price Prediction")
st.subheader("Smartphone Dataset Preview")
st.dataframe(df)  # Show full dataset in Streamlit

# Load model
pkl_path = os.path.join(BASE_DIR, "smartphone_price_model.pkl")
with open(pkl_path, "rb") as f:
    model = pickle.load(f)

st.success("Model loaded successfully! ✅")

# Load and display image
img_path = os.path.join(BASE_DIR, "image.png")
image = Image.open(img_path)
st.image(image, caption="Mobile Price Prediction Project", use_column_width=True)

st.write("You can now use this app to predict mobile prices based on features!")
