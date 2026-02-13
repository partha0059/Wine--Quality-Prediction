import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import ssl

# Fix SSL Certificate Issue
ssl._create_default_https_context = ssl._create_unverified_context

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Pro | Partha Sarathi R",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 12px 40px 0 rgba(114, 47, 55, 0.1);
        margin-bottom: 25px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(114, 47, 55, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(114, 47, 55, 0.15);
    }
    
    h1, h2, h3 {
        color: #722f37;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #722f37 0%, #a03c4a 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 15px 30px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(114, 47, 55, 0.3);
    }
    
    .dev-badge {
        background: rgba(114, 47, 55, 0.1);
        padding: 8px 16px;
        border-radius: 50px;
        color: #722f37;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_best_model():
    try:
        return joblib.load('best_wine_model.pkl')
    except:
        return None

model = load_best_model()

# Sidebar
with st.sidebar:
    st.markdown('<div class="dev-badge">PRO EDITION</div>', unsafe_allow_html=True)
    st.title("🍷 WinePro AI")
    st.markdown("---")
    st.write("This application uses an optimized **Random Forest Regressor** to predict wine quality with high precision.")
    st.markdown("### Developer")
    st.success("👨‍💻 Partha Sarathi R")
    st.markdown("---")
    st.info("The model was trained using 10-Fold Cross-Validation, ensuring maximum robustness and accuracy.")

# Header
st.markdown('<div class="dev-badge">Created by Partha Sarathi R</div>', unsafe_allow_html=True)
st.title("🍇 Premium Wine Quality Analytics")
st.markdown("Experience the future of enology with our AI-driven quality prediction engine.")

if model is None:
    st.error("⚠️ Model not found! Please run the training script first.")
    st.stop()

# Layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Physical-Chemical Analysis")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        fixed_acidity = st.slider("Fixed Acidity", 0.0, 20.0, 7.4)
        volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.7)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    with c2:
        residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
        chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
        free_sd = st.number_input("Free SO2", 0.0, 100.0, 11.0)
    with c3:
        total_sd = st.number_input("Total SO2", 0.0, 300.0, 34.0)
        density = st.number_input("Density", 0.98, 1.01, 0.9978, format="%.4f")
        ph = st.number_input("pH", 2.0, 5.0, 3.51)
        
    sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56)
    alcohol = st.slider("Alcohol Content (%)", 8.0, 15.0, 9.4)
    
    predict_btn = st.button("Analyze & Predict")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_btn:
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sd, total_sd, density, ph, sulphates, alcohol
        ]])
        
        prediction = model.predict(input_data)[0]
        
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.subheader("Results")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Quality Score", 'font': {'size': 24, 'color': '#722f37'}},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#722f37"},
                'bar': {'color': "#722f37"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "rgba(114, 47, 55, 0.1)",
                'steps': [
                    {'range': [0, 5], 'color': 'rgba(255, 87, 87, 0.3)'},
                    {'range': [5, 7], 'color': 'rgba(255, 217, 61, 0.3)'},
                    {'range': [7, 10], 'color': 'rgba(107, 207, 127, 0.3)'}
                ],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=0, b=0, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction >= 7:
            st.balloons()
            st.success("🌟 GRAND CRU QUALITY")
        elif prediction >= 5:
            st.warning("🙂 TABLE QUALITY")
        else:
            st.error("😐 INDUSTRIAL QUALITY")
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="text-align: center; height: 350px; display: flex; align-items: center; justify-content: center;">', unsafe_allow_html=True)
        st.markdown("<p style='color: #722f37; font-size: 18px;'>Ready to analyze.<br>Enter data and click Predict.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #722f37; padding: 40px; font-weight: 600;">
    Developed by Partha Sarathi R © 2024
</div>
""", unsafe_allow_html=True)
