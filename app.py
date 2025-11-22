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
x = pickle.load(open('smartphone_price_model.pkl', 'rb'))


dfen = pd.read_csv('smartphone_cleaned_v1.csv')
dfen = dfen[['brand_name', 'model', 'processor_brand', 'os']]
oe = OrdinalEncoder()
dfen['brand_name_enc'] = oe.fit_transform(dfen[['brand_name']])
dfen['model_enc'] = oe.fit_transform(dfen[['model']])
dfen['processor_brand_enc'] = oe.fit_transform(dfen[['processor_brand']])
dfen['os_enc'] = oe.fit_transform(dfen[['os']])

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

# --------------------------------------
# Sidebar Navigation
# --------------------------------------
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

    # -------------------- Hero Image --------------------
    import base64

    try:
        with open("image1.png", "rb") as img_file:
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

            <!-- -------------------- Page Header -------------------- -->

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
        st.markdown('')
        st.markdown(
        """
        <div style="border-radius:12px; border:2px solid red; padding:20px; background:linear-gradient(135deg, #1a1a1a, #000000); color:white; font-family:'Poppins', sans-serif;">
        <h2 style="text-align:center; color:#FF4500;">Platform Overview</h2>

        <h3 style="color:#FF6347;">About Modern Smartphones</h3>
        <p>
            Smartphones today are more advanced than ever — offering powerful processors, AI features,
            high-resolution cameras, fast charging technology, 5G connectivity, and stunning display quality.
            Whether you're buying a budget device or a flagship phone, each model brings unique features
            designed for performance, photography, gaming, and productivity.
            From AMOLED displays to massive batteries and advanced chipsets, the smartphone market continues
            to evolve rapidly. Learning about these features helps you choose the perfect device based on speed,
            storage, battery life, durability, and camera performance.
        </p>

        <h3 style="color:#FF6347;"> Why This Platform?</h3>
        <p>
            This platform provides smartphone insights, comparisons, and trends based on real market data.
            This homepage introduces you to the world of modern mobile technology. Explore other pages to see
            analysis dashboards, specifications, datasets, and detailed data-driven insights.</p>

        

        <h3 style="color:#FF6347;">Intelligent ML-Powered Tool for Smartphone Pricing</h3>
        <p>
            The Mobile Price Prediction Platform is a smart web-based tool built with machine learning that accurately predicts the price range of any smartphone based on its features like RAM, battery power, camera quality, screen size, and processor speed. Users simply input the phone specifications, and the trained model instantly classifies it into budget, mid-range, high-end, or premium categories. Developed using Python, Scikit-learn, and deployed with Streamlit, it helps buyers make informed decisions and allows mobile companies to position their products competitively in the market.
        </p>

        <p style="text-align:center; color:#5188b5;">
            LetTech AI engineers bring expertise, innovation, and creativity to develop advanced machine learning solutions!
        </p>
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

    # Input Form
    import pandas as pd
    df = pd.read_csv('smartphone_cleaned_v1.csv')
    brand = st.selectbox("Brand Name", df['brand_name'].unique())
    brand1 = dfen[dfen['brand_name']==brand]['brand_name_enc'].iloc[0]

    os = st.selectbox('Operating System', df['os'].unique())
    os1 = dfen[dfen['os']==os]['os_enc'].iloc[0]

    processor_brand = st.selectbox('Processor Brand', df[df['brand_name']== brand]['processor_brand'].unique())
    processor_brand1 = dfen[dfen['processor_brand']==processor_brand]['processor_brand_enc'].iloc[0]

    model = st.selectbox('Model', df[df['brand_name'] == brand]['model'].unique())
    model1 = dfen[dfen['model']==model]['model_enc'].iloc[0]

    ram = st.slider("RAM (GB)", min_value=2, max_value= 16, step=2)
    cores = st.selectbox("Cores", [4, 6, 8])
    storage = st.selectbox("Storage (GB)", [4, 8, 16, 32, 64, 128, 256, 512])
    battery = st.selectbox("Battery (mAh)", df['battery_capacity'].unique())
    number_rear_camera = st.number_input('Number of Rear Cameras', min_value=1, max_value=5, step=1)
    number_front_camera = st.number_input('Number of Front Cameras', 1, 2)
    screen = st.number_input("Screen Size (Inches)", 4.5, 7.5, 6.4)
    rating = df[df['model'] == model]['rating'].mean()
    fast_charge = df[df['model'] == model]['fast_charging'].mean()
    refresh_rate = df[df['model'] == model]['refresh_rate'].mean()
    primary_camera_rear = df[df['model'] == model]['primary_camera_rear'].mean()
    primary_camera_rear = df[df['model'] == model]['primary_camera_rear'].mean()
    extended_memory_available = df[df['model'] == model]['extended_memory_available'].mean()
    extended_upto = df[df['model'] == model]['extended_upto'].mean()
    resolution_width = df[df['model'] == model]['resolution_width'].mean()
    resolution_height = df[df['model'] == model]['resolution_height'].mean()

    

    processor_speed = st.number_input("Processor Speed (GHz)", 1.0, 3.5, 2.2)


    col1, col2, col3 = st.columns(3)
    fiveG = col1.checkbox("5G Support")
    if fiveG== True:
        fiveG=1
    else:
        fiveG=0


    nfc = col2.checkbox("NFC")
    if nfc== True:
        nfc=1
    else:
        nfc=0

    ir = col3.checkbox("IR Blaster")
    if ir== True:
        ir=1
    else:
        ir=0

    if st.button("Predict Price"):


        price = x.predict([[brand1, model1, rating, fiveG, nfc, ir,
        processor_brand1, cores, processor_speed, battery,
        fast_charge,fast_charge, ram,
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
    import base64
    with open("image.png", "rb") as img_file:
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

            <!-- -------------------- Page Header -------------------- -->

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
        
    # app.py
    import streamlit as st
    from PIL import Image

    # Page configuration
    st.set_page_config(
        page_title="Mobile Price Prediction",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load project image
    image = Image.open("image.png")
    st.markdown('')
    # Main container
    st.markdown(
        """
        <div style="border-radius:12px; border:2px solid red; padding:20px; background:linear-gradient(135deg, #000000, #000000); color:white; font-family:'Poppins', sans-serif;"> <h2 style="text-align:center; color:#8B0000;">Mobile Price Prediction - Extra Trees Regressor</h2> <h3 style="color:#8B0000;">Introduction</h3> <p> This model predicts the <strong>price of smartphones</strong> based on multiple features like RAM, Battery Capacity, Processor, Screen Size, Cameras, and more. It is built using the <strong>Extra Trees Regressor</strong> algorithm and deployed via <strong>Streamlit</strong> for an interactive, user-friendly interface. </p> <h3 style="color:#8B0000;">Why Extra Trees Regressor?</h3> <ul> <li><strong>High Accuracy:</strong> Handles high-dimensional data effectively.</li> <li><strong>Robustness:</strong> Reduces overfitting using randomization across trees.</li> <li><strong>Fast Training:</strong> Works well on large datasets.</li> <li><strong>Handles Non-linear Relationships:</strong> Excellent for complex regression data.</li> </ul> <h3 style="color:#8B0000;">Performance</h3> <p> The model achieves an impressive <strong>R² Score of 0.8851</strong> (≈88.51%) on the test data. <strong>Mean Absolute Error (MAE):</strong> 2907  <br> <strong>Mean Squared Error (MSE):</strong> 18,612,758 </p> <h3 style="color:#8B0000;">Features Included</h3> <p> * RAM, Internal Storage, Battery Capacity<br> * Processor Brand, Processor Speed, Number of Cores<br> * Screen Size, Refresh Rate, Resolution<br> * Front & Rear Camera Specifications<br> * Connectivity features: 5G, NFC, IR Blaster </p> <h3 style="color:#8B0000;">Streamlit Deployment</h3> <p> Users can input smartphone specifications and instantly get <strong>predicted prices</strong> in PKR. The app includes an interactive UI with sliders, dropdowns, and colorful charts for an engaging experience. </p> <p style="text-align:center; font-weight:bold; color:#5188b5;">LetTech AI engineers bring expertise, innovation, and creativity to develop advanced machine learning solutions! </p> </div>
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
        <p>
            My name is Musa Khan, and I am a professional AI Engineer specializing in machine learning, deep learning, and data-driven solutions. I work with Python and modern AI frameworks to build intelligent systems that solve real-world problems. My expertise includes model development, data preprocessing, visualization, and deploying AI models for practical use.
            I am passionate about creating advanced AI applications, including systems that can think, learn, and assist in everyday tasks. I continuously expand my skills in areas like computer vision, agentic AI, and automation to stay ahead in the rapidly evolving field of artificial intelligence. With a strong focus on innovation and continuous learning, I aim to build impactful AI technologies that contribute to a smarter future.


        </p>

        <h3 style="color:#FF6347;">Miraj Ud Din</h3>
        <p>Miraj Ud Din is a BS (AI) student from Peshawar, specializing in Data Science, Machine Learning, Deep Learning and Artificial Intelligence.
        He has strong expertise in Python programming and SQL.
        Miraj focuses on developing intelligent systems and deploying machine learning models effectively.
        He enjoys creating interactive web applications that integrate AI solutions.
        He applies practical examples relevant to the World context in his projects and solutions.
        Miraj is passionate about advancing AI technologies and making them accessible for real-world applications.</p>

        

        <h3 style="color:#FF6347;">Ahmad Aziz</h3>
        <p>
            Ahmad Aziz is an experienced AI engineer specializing in deep learning and data-driven solutions.
            He excels in building and deploying machine learning applications with high performance.
            His expertise includes creating interactive user interfaces for AI tools.
            He ensures seamless integration of ML models into web and desktop platforms.
            Ahmad is passionate about practical, real-world AI implementations.
            He helps make advanced technology accessible and user-friendly.
        </p>

        <p style="text-align:center; color:#5188b5;">
            LetTech AI engineers bring expertise, innovation, and creativity to develop advanced machine learning solutions!
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )




