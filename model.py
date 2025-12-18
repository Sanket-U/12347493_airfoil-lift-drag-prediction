import torch
import torch.nn as nn

class AirfoilMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for aerodynamic coefficient regression.
    
    Architecture designed to capture non-linear relationships between 
    airfoil geometry and aerodynamic forces (Lift/Drag).
    
    Structure: Input(102) -> 256 -> 128 -> 64 -> Output(2)
    """
    def __init__(self, input_size=102):
        super(AirfoilMLP, self).__init__()
        
        self.network = nn.Sequential(
            # --- Hidden Layer 1 ---
            # Expansion layer to capture high-level geometric features
            nn.Linear(input_size, 256),
            nn.ReLU(),              
            nn.BatchNorm1d(256),    # Batch Normalization for training stability
            
            # --- Hidden Layer 2 ---
            # Compression layer
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # --- Hidden Layer 3 ---
            # Feature refinement before output
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # --- Output Layer ---
            # Regression output: [Lift (Cl), Drag (Cd)]
            nn.Linear(64, 2) 
        )
        
    def forward(self, x):
        return self.network(x)