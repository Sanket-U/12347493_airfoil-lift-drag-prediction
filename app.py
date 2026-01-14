import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from model import AirfoilMLP
from dataset import AirfoilDataset

#CONFIGURATION
st.set_page_config(page_title="Airfoil Deep Learning Demo", layout="wide")
st.title("Applied Deep Learning: Airfoil Surrogate Model")

#HELPER FUNCTIONS
def interpolate_to_50_points(x, y):
    dist = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    dist = dist / dist[-1]
    fx = interp1d(dist, x, kind='linear')
    fy = interp1d(dist, y, kind='linear')
    alpha = np.linspace(0, 1, 50)
    return fx(alpha), fy(alpha)

@st.cache_resource
def load_resources():
    model = AirfoilMLP()
    model.load_state_dict(torch.load('airfoil_model_smart.pth', map_location=torch.device('cpu')))
    model.eval()
    ds = AirfoilDataset('DeepLearWing.csv', subset=1000, fixed_points=50)
    return model, ds

try:
    model, dataset = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

#LAYOUT
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("1. Geometry Input")
    source = st.radio("Source:", ["Dataset Sample", "Upload CSV (Unseen Data)"])
    
    current_geometry = None
    
    if source == "Dataset Sample":
        sample_id = st.number_input("Sample ID", 0, 999, 0)
        raw_sample, _ = dataset[sample_id]
        current_geometry = raw_sample[:100]
        st.info(f"Loaded Sample #{sample_id}")
    
    else: # Upload
        uploaded_file = st.file_uploader("Upload Airfoil CSV", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                x_raw, y_raw = df[0].values, df[1].values
                x_50, y_50 = interpolate_to_50_points(x_raw, y_raw)
                current_geometry = torch.tensor(np.concatenate([x_50, y_50]), dtype=torch.float32)
                st.success("Geometry Processed")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.header("2. Flow Conditions")

    #aoa
    col1, col2 = st.columns([3, 1]) 
    with col1:
        #slider
        aoa_slide = st.slider("Angle of Attack (°)", -20.0, 20.0, 0.0, 0.5, key='aoa_slider')
    with col2:
        #Input box 
        aoa = st.number_input("Precise AoA", value=float(aoa_slide), step=0.1, label_visibility="collapsed")

    #Re
    col3, col4 = st.columns([3, 1])
    with col3:
        #Slider
        re_slide = st.slider("Reynolds Number", 50000, 1000000, 200000, 10000, key='re_slider')
    with col4:
        #Input box
        reynolds = st.number_input("Precise Re", value=int(re_slide), step=1000, label_visibility="collapsed")

# PREDICTION
if current_geometry is not None:
    # Prepare Input
    aoa_scaled = aoa / 20.0
    re_scaled = np.log10(reynolds) / 6.0
    physics = torch.tensor([aoa_scaled, re_scaled], dtype=torch.float32)
    full_input = torch.cat((current_geometry, physics))

    # Measure Inference Speed
    start_time = time.time()
    with torch.no_grad():
        pred = model(full_input.unsqueeze(0))
        cl_pred = pred[0][0].item()
        cd_pred = pred[0][1].item()
    end_time = time.time()
    inference_time = end_time - start_time

    # VISUALIZATION
    with right_col:
        st.subheader("Geometry Visualization")
        x_plot = current_geometry[:50].numpy()
        y_plot = current_geometry[50:].numpy()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill(x_plot, y_plot, color='#1f77b4', alpha=0.6)
        ax.plot(x_plot, y_plot, 'k-', lw=2)
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Flow Arrow
        rad = np.radians(aoa)
        ax.arrow(-0.2, 0, 0.3*np.cos(rad), 0.3*np.sin(rad), width=0.01, color='red', label='Flow')
        ax.legend()
        st.pyplot(fig)

        # Metrics
        st.subheader("Neural Network Prediction")
        c1, c2 = st.columns(2)
        c1.metric("Lift ($C_l$)", f"{cl_pred:.4f}")
        c2.metric("Drag ($C_d$)", f"{cd_pred:.5f}")
        
        st.caption(f"Inference Time: {inference_time:.6f} sec")