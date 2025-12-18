import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AirfoilDataset
from model import AirfoilMLP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
BATCH_SIZE = 1024
MODEL_PATH = "airfoil_model_smart.pth"

def evaluate():
    """
    Loads the trained model and evaluates performance on a test subset.
    Calculates R2 and MAE metrics and generates regression scatter plots.
    """
    print("--- STARTING EVALUATION ---")
    
    # 1. Load Test Data
    # Using a random subset of 5000 samples for visualization clarity
    dataset = AirfoilDataset('DeepLearWing.csv', subset=5000, fixed_points=50)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # 2. Initialize Model
    model = AirfoilMLP(input_size=102)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Set to inference mode
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    # 3. Inference Loop
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            all_preds.append(outputs.numpy())
            all_targets.append(targets.numpy())
            
    # Stack batches into single numpy arrays
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # 4. Metric Calculation
    # Target indices: 0 = Lift (Cl), 1 = Drag (Cd)
    cl_true, cl_pred = targets[:, 0], preds[:, 0]
    cd_true, cd_pred = targets[:, 1], preds[:, 1]
    
    r2_cl = r2_score(cl_true, cl_pred)
    r2_cd = r2_score(cd_true, cd_pred)
    
    mae_cl = mean_absolute_error(cl_true, cl_pred)
    mae_cd = mean_absolute_error(cd_true, cd_pred)
    
    print("\n" + "="*30)
    print("FINAL PERFORMANCE METRICS")
    print("="*30)
    print(f"LIFT (Cl) -> R2: {r2_cl:.4f} | MAE: {mae_cl:.4f}")
    print(f"DRAG (Cd) -> R2: {r2_cd:.4f} | MAE: {mae_cd:.4f}")
    print("="*30)

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Lift Plot
    axes[0].scatter(cl_true, cl_pred, alpha=0.3, s=5, color='blue')
    # Draw ideal 1:1 line
    axes[0].plot([cl_true.min(), cl_true.max()], [cl_true.min(), cl_true.max()], 'r--', lw=2)
    axes[0].set_title(f"Lift Prediction ($R^2$={r2_cl:.2f})")
    axes[0].set_xlabel("True Cl (Physics)")
    axes[0].set_ylabel("Predicted Cl (Neural Net)")
    axes[0].grid(True)
    
    # Drag Plot
    axes[1].scatter(cd_true, cd_pred, alpha=0.3, s=5, color='green')
    axes[1].plot([cd_true.min(), cd_true.max()], [cd_true.min(), cd_true.max()], 'r--', lw=2)
    axes[1].set_title(f"Drag Prediction ($R^2$={r2_cd:.2f})")
    axes[1].set_xlabel("True Cd (Physics)")
    axes[1].set_ylabel("Predicted Cd (Neural Net)")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("Visualization saved to 'evaluation_results.png'")

if __name__ == "__main__":
    evaluate()