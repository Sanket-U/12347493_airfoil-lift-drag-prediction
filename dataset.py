import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AirfoilDataset(Dataset):
    """
    PyTorch Dataset for aerodynamic coefficient prediction.
    
    Loads airfoil geometry and flow conditions from CSV, applies data standardization, 
    and resamples variable-length geometry strings to a fixed coordinate size.
    """
    def __init__(self, csv_file, subset=None, fixed_points=50):
        """
        Args:
            csv_file (str): Path to the source CSV file.
            subset (int, optional): Number of samples to load (for debugging).
            fixed_points (int): Number of points to resample airfoil geometry to.
        """
        print(f"Loading data from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        
        if subset:
            print(f"Using a subset of {subset} samples for testing!")
            self.data = self.data.iloc[:subset]

        print(f"Resampling all airfoils to {fixed_points} points...")
        self.x_coords = []
        self.y_coords = []
        
        # Resample geometry to ensure fixed input vector size
        for i in range(len(self.data)):
            # Parse space-separated coordinate strings to numpy arrays
            raw_x = np.fromstring(self.data.iloc[i]['x_coords'], sep=' ')
            raw_y = np.fromstring(self.data.iloc[i]['y_coords'], sep=' ')
            
            # Linear interpolation to standardize point count
            new_indices = np.linspace(0, 1, fixed_points)
            old_indices = np.linspace(0, 1, len(raw_x))
            
            new_x = np.interp(new_indices, old_indices, raw_x)
            new_y = np.interp(new_indices, old_indices, raw_y)
            
            self.x_coords.append(new_x)
            self.y_coords.append(new_y)

        # Convert lists to efficient numpy matrices
        self.x_coords = np.array(self.x_coords, dtype=np.float32)
        self.y_coords = np.array(self.y_coords, dtype=np.float32)

        # --- Physics Input Normalization ---
        # Angle of Attack: Scale [-20, 20] -> [-1.0, 1.0]
        self.angles = self.data['angle'].values.astype(np.float32) / 20.0
        
        # Reynolds Number: Log-transform to compress order of magnitude [50k, 1M]
        self.reynolds = np.log10(self.data['reynolds'].values.astype(np.float32)) / 6.0
        
        # Targets
        self.cl = self.data['cl'].values.astype(np.float32)
        self.cd = self.data['cd'].values.astype(np.float32)
        
        # Input dimension: (x_coords + y_coords) + angle + reynolds
        self.input_dim = (fixed_points * 2) + 2
        
        print(f"Dataset Loaded. Samples: {len(self.data)}")
        print(f"   Input Features: {self.input_dim} (Standardized)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Concatenate geometry and physics into single input vector
        geo = np.concatenate([self.x_coords[idx], self.y_coords[idx]])
        phys = np.array([self.angles[idx], self.reynolds[idx]], dtype=np.float32)
        
        x = np.concatenate([geo, phys])
        y = np.array([self.cl[idx], self.cd[idx]], dtype=np.float32)
        
        return torch.tensor(x), torch.tensor(y)