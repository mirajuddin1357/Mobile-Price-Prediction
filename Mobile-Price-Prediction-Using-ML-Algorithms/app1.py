import os
import pickle
import pandas as pd
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(BASE_DIR, "smartphone_cleaned_v1.csv"))
with open(os.path.join(BASE_DIR, "smartphone_price_model.pkl"), "rb") as f:
    model = pickle.load(f)
img = Image.open(os.path.join(BASE_DIR, "image.png"))
st.image(img, width=600)
