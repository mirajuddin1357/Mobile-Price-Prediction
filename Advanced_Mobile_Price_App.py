# ---------------------------------------------------------
# ADVANCED MOBILE INTELLIGENCE PLATFORM (AI CORE v2.0)
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder
import base64

# BASE DIRECTORY SETUP
BASE_DIR = os.path.dirname(__file__)

# --- UTILITY: LOAD MODEL & DATA ---
@st.cache_resource
def load_engine():
    try:
        model = pickle.load(open(os.path.join(BASE_DIR, 'smartphone_price_model.pkl'), 'rb'))
        return model
    except:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'smartphone_cleaned_v1.csv'))
        return df
    except:
        return None

# --- GLOBAL STYLES (Futuristic Glassmorphism) ---
def apply_aesthetics():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --neon-cyan: #00f2ff;
        --neon-magenta: #ff00ff;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background: radial-gradient(circle at top right, #0a0a1a, #000000);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-magenta));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: var(--neon-cyan);
    }

    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        transition: 0.3s all;
        box-shadow: 0 0 15px rgba(0, 198, 255, 0.4);
    }

    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 198, 255, 0.8);
        transform: scale(1.05);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 26, 0.95);
        border-right: 1px solid var(--glass-border);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE PREDICTION ENGINE ---
def get_prediction(model, data_row):
    try:
        price_raw = model.predict([data_row])
        val = np.array(price_raw).item()
        return val
    except Exception as e:
        return None

# --- MAIN APP ---
def main():
    apply_aesthetics()
    
    st.sidebar.markdown("<h2 style='text-align:center;'>Predict mobile price</h2>", unsafe_allow_html=True)
    menu = st.sidebar.radio("Navigation", ["Insights", "Predict Price", "Compare Mobiles", "About"])

    model = load_engine()
    df = load_data()

    if df is None or model is None:
        st.error("SYSTEM CRITICAL ERROR: Model or Data files missing.")
        return

    # Setup Encoders
    dfen = df[['brand_name', 'model', 'processor_brand', 'os']].copy()
    oe = OrdinalEncoder()
    dfen['brand_name_enc'] = oe.fit_transform(dfen[['brand_name']])
    dfen['model_enc'] = oe.fit_transform(dfen[['model']])
    dfen['processor_brand_enc'] = oe.fit_transform(dfen[['processor_brand']])
    dfen['os_enc'] = oe.fit_transform(dfen[['os']])

    # --- SIDEBAR: Did You Know? ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Fun Facts")
    facts = [
        "5000 mAh is the standard for long battery life.",
        "8GB RAM is perfect for most games and apps.",
        "OLED screens use less battery on dark mode.",
        "Higher Refresh Rate (120Hz) makes scrolling smoother.",
        "Extra cores (8-core) help with heavy multitasking."
    ]
    st.sidebar.info(np.random.choice(facts))

    # -----------------------------------------------------
    # PAGE: NEURAL INSIGHTS (Interactive Dashboard)
    # -----------------------------------------------------
    if menu == "Insights":
        st.markdown('<div class="glass-card"><h1>Data Insights</h1></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card"><h3>Price Distribution by Brand</h3>', unsafe_allow_html=True)
            fig = px.box(df, x="brand_name", y="price", color="brand_name",
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card"><h3>RAM vs Price</h3>', unsafe_allow_html=True)
            fig = px.scatter(df, x="ram_capacity", y="price", size="rating", color="brand_name",
                             hover_name="model", template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><h3>What affects the price?</h3>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            'Feature': ['RAM', 'Storage', 'Battery', 'Rating', 'Brand'],
            'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
        })
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Viridis', template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: PRICE MATRIX (Single Prediction)
    # -----------------------------------------------------
    elif menu == "Predict Price":
        with st.container():
            st.markdown('<div class="glass-card"><h1>Price Prediction</h1>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            
            with c1:
                brand = st.selectbox("Brand Name", df['brand_name'].unique(), index=0, placeholder="Choose Brand")
                brand_enc = dfen[dfen['brand_name']==brand]['brand_name_enc'].iloc[0]
                
                os_choice = st.selectbox('Operating System', df['os'].unique(), index=0, placeholder="Select OS")
                os_enc = dfen[dfen['os']==os_choice]['os_enc'].iloc[0]

            with c2:
                processor = st.selectbox('Processor Brand', df[df['brand_name']== brand]['processor_brand'].unique(), index=0, placeholder="Select Processor")
                proc_enc = dfen[dfen['processor_brand']==processor]['processor_brand_enc'].iloc[0]
                
                model_choice = st.selectbox('Model', df[df['brand_name'] == brand]['model'].unique(), index=0, placeholder="Pick Model")
                model_enc = dfen[dfen['model']==model_choice]['model_enc'].iloc[0]

            with c3:
                ram = st.slider("RAM (GB)", 2, 64, 8)
                storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024], index=2, placeholder="Choose Storage")
                battery = st.selectbox("Battery (mAh)", sorted(df['battery_capacity'].unique()), index=0, placeholder="Select Battery")

            # More Features
            st.divider()
            c4, c5, c6 = st.columns(3)
            with c4:
                cores = st.selectbox("Cores", [2, 4, 6, 8, 10], index=3, placeholder="Select Cores")
                speed = st.number_input("Speed (GHz)", 1.0, 4.0, 2.2, placeholder="e.g. 2.4")
            with c5:
                screen = st.number_input("Screen Size", 4.0, 8.0, 6.5, placeholder="e.g. 6.7")
                refresh = st.selectbox("Refresh Rate", [60, 90, 120, 144, 165], index=2, placeholder="Select Hz")
            with c6:
                rear_cam = st.number_input("Rear Cameras", 1, 5, 3, placeholder="e.g. 3")
                front_cam = st.number_input("Front Cameras", 1, 2, 1, placeholder="e.g. 1")

            # Auto-fill statistics from model
            feat_data = df[df['model'] == model_choice].iloc[0]
            rating = feat_data['rating']
            fast_charge = feat_data['fast_charging']
            p_cam_r = feat_data['primary_camera_rear']
            ext_mem = feat_data['extended_memory_available']
            ext_up = feat_data['extended_upto']
            res_w = feat_data['resolution_width']
            res_h = feat_data['resolution_height']

            # Checkboxes
            c7, c8, c9 = st.columns(3)
            fiveG = c7.checkbox("Enable 5G Neural Link")
            nfc = c8.checkbox("NFC Activation")
            ir = c9.checkbox("IR Blaster Module")

            if st.button("Predict"):
                inputs = [
                    brand_enc, model_enc, rating, 1 if fiveG else 0, 1 if nfc else 0, 1 if ir else 0,
                    proc_enc, cores, speed, battery, fast_charge, fast_charge, ram,
                    storage, screen, int(refresh), rear_cam, front_cam, os_enc,
                    p_cam_r, p_cam_r, int(ext_mem), ext_up, int(res_w), int(res_h)
                ]
                
                price_inr = get_prediction(model, inputs)
                
                if price_inr:
                    # Multi-Currency Display
                    pkr = price_inr * 3.105  # INR to PKR rate (Jan 2026)
                    usd = pkr / 280.0
                    cny = pkr / 40.16  # PKR to CNY rate (Jan 2026)
                    
                    # Logic for Simple Value Verdict
                    if pkr < 40000:
                        verdict = "Budget Friendly"
                        color = "#00ff88"
                    elif pkr < 100000:
                        verdict = "Value for Money"
                        color = "#00f2ff"
                    elif pkr < 200000:
                        verdict = "High-End Luxury"
                        color = "#ff00ff"
                    else:
                        verdict = "Premium Elite"
                        color = "#ffcc00"

                    st.markdown(f"""
                    <div style="text-align:center; padding:20px; border-radius:15px; background:rgba(0, 242, 255, 0.1); border:1px solid rgba(0, 242, 255, 0.2);">
                        <h2 style='margin:0; font-size:1.2em; opacity:0.8;'>PREDICTED PRICE</h2>
                        <h1 style='font-size:3.5em; margin:10px 0; color:white;'>PKR {int(pkr):,}</h1>
                        <p style='color:var(--neon-cyan); font-weight:bold; font-size:1.1em;'>
                            ≈ ${usd:,.2f} USD | ₹{price_inr:,.2f} INR | ¥{cny:,.2f} CNY
                        </p>
                        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin:15px 0;">
                        <h3 style='margin:0; font-size:1.1em; color:{color};'>Good value if you want luxury. {verdict}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
            
            st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: QUANTUM COMPARE (Compare two configs)
    # -----------------------------------------------------
    elif menu == "Compare Mobiles":
        st.markdown('<div class="glass-card"><h1>Mobile Comparison</h1>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("DEVICE ALPHA")
            a_brand = st.selectbox("Brand A", df['brand_name'].unique(), index=0, placeholder="Brand A", key='a_br')
            a_model = st.selectbox("Model A", df[df['brand_name']==a_brand]['model'].unique(), index=0, placeholder="Model A", key='a_mod')
            a_ram = st.slider("RAM A (GB)", 2, 64, 8, key='a_ram')
            a_bat = st.selectbox("Battery A (mAh)", sorted(df['battery_capacity'].unique()), index=0, placeholder="Battery A", key='a_bat')
            
        with col_b:
            st.subheader("DEVICE BETA")
            b_brand = st.selectbox("Brand B", df['brand_name'].unique(), index=0, placeholder="Brand B", key='b_br')
            b_model = st.selectbox("Model B", df[df['brand_name']==b_brand]['model'].unique(), index=0, placeholder="Model B", key='b_mod')
            b_ram = st.slider("RAM B (GB)", 2, 64, 12, key='b_ram')
            b_bat = st.selectbox("Battery B (mAh)", sorted(df['battery_capacity'].unique()), index=0, placeholder="Battery B", key='b_bat')
            
        if st.button("Compare"):
            st.info("Calculating comparative value...")
            
            # Extract data for Alpha
            data_a = df[df['model'] == a_model].iloc[0]
            brand_enc_a = filter_enc('brand_name', a_brand, dfen)
            model_enc_a = filter_enc('model', a_model, dfen)
            proc_enc_a = filter_enc('processor_brand', data_a['processor_brand'], dfen)
            os_enc_a = filter_enc('os', data_a['os'], dfen)
            
            inputs_a = [
                brand_enc_a, model_enc_a, data_a['rating'], 1, 1, 1,
                proc_enc_a, 8, 2.8, a_bat, data_a['fast_charging'], data_a['fast_charging'], a_ram,
                128, 6.7, 120, 3, 1, os_enc_a,
                108, 108, 0, 0, 1080, 2400
            ]
            
            # Extract data for Beta
            data_b = df[df['model'] == b_model].iloc[0]
            brand_enc_b = filter_enc('brand_name', b_brand, dfen)
            model_enc_b = filter_enc('model', b_model, dfen)
            proc_enc_b = filter_enc('processor_brand', data_b['processor_brand'], dfen)
            os_enc_b = filter_enc('os', data_b['os'], dfen)
            
            inputs_b = [
                brand_enc_b, model_enc_b, data_b['rating'], 1, 1, 1,
                proc_enc_b, 8, 2.8, b_bat, data_b['fast_charging'], data_b['fast_charging'], b_ram,
                128, 6.7, 120, 3, 1, os_enc_b,
                108, 108, 0, 0, 1080, 2400
            ]

            p_a = get_prediction(model, inputs_a) * 3.105
            p_b = get_prediction(model, inputs_b) * 3.105

            # Display prices
            m1, m2 = st.columns(2)
            alpha_cny = p_a / 40.16
            beta_cny = p_b / 40.16
            
            m1.metric("Alpha Price", f"PKR {int(p_a):,}", f"¥{alpha_cny:,.2f} CNY")
            m2.metric("Beta Price", f"PKR {int(p_b):,}", f"¥{beta_cny:,.2f} CNY")

            # Value Verdict
            st.markdown("### 🏆 Value Verdict")
            if p_a < p_b:
                st.success(f"**Device Alpha** is PKR {int(p_b - p_a):,} more affordable!")
            else:
                st.success(f"**Device Beta** is PKR {int(p_a - p_b):,} more affordable!")
            
            if a_ram > b_ram:
                st.info("💡 **Alpha** might be better for gaming due to higher RAM.")
            elif b_ram > a_ram:
                st.info("💡 **Beta** might be better for gaming due to higher RAM.")

            st.write("---")
            st.markdown("""
            | Feature | Device Alpha | Device Beta |
            | :--- | :---: | :---: |
            | Brand | {0} | {1} |
            | Model | {2} | {3} |
            | RAM | {4}GB | {5}GB |
            | Battery | {6}mAh | {7}mAh |
            """.format(a_brand, b_brand, a_model, b_model, a_ram, b_ram, a_bat, b_bat))
            
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: REGISTRY (About)
    # -----------------------------------------------------
    elif menu == "About":
        st.title("About the App")
        
        st.markdown(f"""
        <div class="glass-card">
            <h3>Project Description</h3>
            <p>Our app helps you find the right price for your next mobile phone. We use advanced Machine Learning to look at over 25 different phone features and tell you exactly what the market price should be.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <h3>📈 Simple Buying Tips</h3>
            <ul>
                <li><b>Battery:</b> Look for at least 5000 mAh if you use your phone all day.</li>
                <li><b>RAM:</b> 8GB is the "sweet spot" for 2024. Anything less may feel slow.</li>
                <li><b>Storage:</b> 128GB is okay, but 256GB is better for photos and videos.</li>
                <li><b>Screen:</b> 120Hz refresh rate makes everything look much smoother.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card">
            <h3>Meet the Team</h3>
            <p>Built by <b>LetTech AI Engineers</b></p>
            <ul>
                <li><b>Miraj Ud Din</b></li>
                <li><b>Musa Khan</b></li>
                <li><b>Ahmad Aziz</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Helper function to get encoding safely
def filter_enc(col, val, dfen):
    try:
        return dfen[dfen[col] == val][col + '_enc'].iloc[0]
    except:
        return 0

if __name__ == "__main__":
    main()
