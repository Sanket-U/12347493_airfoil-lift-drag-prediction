import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from model import AirfoilMLP

MODEL_PATH = 'airfoil_model_smart.pth'
REYNOLDS = 200000

model = AirfoilMLP()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit()

def interpolate_to_50_points(x, y):
    dist = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    dist = dist / dist[-1]
    fx = interp1d(dist, x, kind='linear')
    fy = interp1d(dist, y, kind='linear')
    alpha = np.linspace(0, 1, 50)
    return fx(alpha), fy(alpha)

def load_geometry(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)
        x_raw, y_raw = df[0].values, df[1].values
        x_50, y_50 = interpolate_to_50_points(x_raw, y_raw)
        return torch.tensor(np.concatenate([x_50, y_50]), dtype=torch.float32)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return None

def predict_polar(geometry_tensor, start_angle, end_angle):
    angles = np.arange(start_angle, end_angle + 0.5, 0.5)
    re_scaled = np.log10(REYNOLDS) / 6.0
    pred_cl, pred_cd = [], []
    
    with torch.no_grad():
        for alpha in angles:
            aoa_scaled = alpha / 20.0
            physics = torch.tensor([aoa_scaled, re_scaled], dtype=torch.float32)
            full_input = torch.cat((geometry_tensor, physics))
            out = model(full_input.unsqueeze(0))
            pred_cl.append(out[0][0].item())
            pred_cd.append(out[0][1].item())
            
    return angles, pred_cl, pred_cd

def plot_save(filename, title, pred_data, xfoil_data):
    angles, pred_cl, pred_cd = pred_data
    xf_alpha, xf_cl, xf_cd = xfoil_data
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(angles, pred_cl, 'b-', linewidth=2, label='Deep Learning (Ours)')
    ax1.plot(xf_alpha, xf_cl, 'r--', linewidth=2, label='XFOIL (Ground Truth)')
    ax1.set_title(f'{title}: Lift Polar (Re={REYNOLDS})')
    ax1.set_xlabel('Angle of Attack (°)')
    ax1.set_ylabel('Lift Coefficient ($C_l$)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(angles, pred_cd, 'b-', linewidth=2, label='Deep Learning (Ours)')
    ax2.plot(xf_alpha, xf_cd, 'r--', linewidth=2, label='XFOIL (Ground Truth)')
    ax2.set_title(f'{title}: Drag Polar (Re={REYNOLDS})')
    ax2.set_xlabel('Angle of Attack (°)')
    ax2.set_ylabel('Drag Coefficient ($C_d$)')
    ax2.set_ylim(0, 0.20)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Graph generated: {filename}")

# NACA 6409
geo_6409 = load_geometry('naca6409.csv')
if geo_6409 is not None:
    data_6409 = predict_polar(geo_6409, -10, 20)
    
    xf_alpha_6409 = [-9.25, -9.0, -8.75, -8.25, -7.75, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 19.25]
    xf_cl_6409 = [-0.2185, -0.2076, -0.1934, -0.1720, -0.1381, -0.0972, -0.0059, 0.1032, 0.2159, 0.3290, 0.4407, 0.5488, 0.6544, 0.7601, 0.8652, 0.9659, 1.0658, 1.1643, 1.2603, 1.3310, 1.3578, 1.3524, 1.3697, 1.3834, 1.3912, 1.3889, 1.4003, 1.4121, 1.4215, 1.4114, 1.3835, 1.3376, 1.3209]
    xf_cd_6409 = [0.10530, 0.10160, 0.09740, 0.08756, 0.08078, 0.06681, 0.04660, 0.01982, 0.01572, 0.01350, 0.01226, 0.01171, 0.01145, 0.01124, 0.01083, 0.01160, 0.01246, 0.01339, 0.01442, 0.01567, 0.01865, 0.02490, 0.03079, 0.03802, 0.04701, 0.05844, 0.06929, 0.08080, 0.09318, 0.10984, 0.13085, 0.15808, 0.16694]
    
    plot_save('naca6409_polar.png', 'NACA 6409', data_6409, (xf_alpha_6409, xf_cl_6409, xf_cd_6409))

# NACA 2414
geo_2414 = load_geometry('naca2414.csv')
if geo_2414 is not None:
    data_2414 = predict_polar(geo_2414, -16, 20)
    
    xf_alpha_2414 = [-15.5, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
    xf_cl_2414    = [-0.8947, -0.9846, -0.9055, -0.6276, -0.3152, -0.0320, 0.2283, 0.5356, 0.8017, 1.0030, 1.1577, 1.2542, 1.2832, 1.2561]
    xf_cd_2414    = [0.08030, 0.03332, 0.02457, 0.01837, 0.01423, 0.01160, 0.01022, 0.01125, 0.01311, 0.01685, 0.02340, 0.03540, 0.05898, 0.09589]
    
    plot_save('naca2414_polar.png', 'NACA 2414', data_2414, (xf_alpha_2414, xf_cl_2414, xf_cd_2414))