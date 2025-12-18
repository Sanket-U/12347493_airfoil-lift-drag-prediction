import matplotlib.pyplot as plt
import numpy as np
from dataset import AirfoilDataset

def plot_geometry_schematic():
    """
    Generates a schematic visualization of the input data structure.
    
    Plots:
    1. The 50 discrete coordinate points (Input features 0-99).
    2. The flow direction based on Angle of Attack (Input feature 100).
    3. Physics labels including Reynolds number (Input feature 101).
    """
    print("--- GENERATING INPUT SCHEMATIC ---")
    
    # Load 3 diverse samples for visualization
    # We use specific indices [0, 10, 55] known to show variety
    ds = AirfoilDataset('DeepLearWing.csv', subset=100, fixed_points=50)
    indices = [0, 10, 55]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, idx in enumerate(indices):
        inputs, targets = ds[idx]
        
        # --- Decode Inputs ---
        # 1. Geometry: First 100 features (50 x-coords, 50 y-coords)
        geo_vector = inputs[:-2].numpy()
        x_coords = geo_vector[:50]
        y_coords = geo_vector[50:]
        
        # 2. Physics: Last 2 features (Angle, Reynolds)
        phys_vector = inputs[-2:].numpy()
        
        # Reverse scaling for display purposes
        real_angle = phys_vector[0] * 20.0 
        real_reynolds = 10 ** (phys_vector[1] * 6.0)
        
        # 3. Targets: Lift and Drag
        cl = targets[0].item()
        cd = targets[1].item()
        
        # --- Plotting ---
        ax = axes[i]
        
        # Plot Input Nodes
        ax.scatter(x_coords, y_coords, color='blue', s=15, label='Input Nodes (50)')
        ax.plot(x_coords, y_coords, color='black', alpha=0.3, linestyle='--')
        
        # Visualize Angle of Attack (Flow Direction)
        # Calculates a vector component based on the angle
        ax.arrow(-0.1, 0, 0.2, np.sin(np.radians(real_angle))*0.2, 
                 head_width=0.02, color='red', alpha=0.6, label='Flow Direction')
        
        # Annotations
        ax.set_title(f"Sample #{idx}: {real_angle:.1f}° AoA", fontsize=12, fontweight='bold')
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axis('equal') 
        ax.set_ylim(-0.2, 0.3)
        
        # Physics Info Box
        info_text = (f"Re: {real_reynolds:.0e}\n"
                     f"Lift ($C_l$): {cl:.2f}\n"
                     f"Drag ($C_d$): {cd:.4f}")
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        if i == 0:
            ax.legend(loc='lower right')

    plt.suptitle("Schematic: Mapping Input Vector (102 Features) to Physical Geometry", fontsize=14)
    plt.tight_layout()
    plt.savefig("geometry_schematic.png")
    print("Schematic saved to 'geometry_schematic.png'")

if __name__ == "__main__":
    plot_geometry_schematic()