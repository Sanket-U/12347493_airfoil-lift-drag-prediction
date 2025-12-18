import unittest
import torch
import numpy as np
from dataset import AirfoilDataset

class TestAirfoilPipeline(unittest.TestCase):
    
    def test_resampling_shape(self):
        """Test if the dataset correctly resamples inputs to 102 features."""
        # We test with the subset mode to load quickly
        ds = AirfoilDataset('DeepLearWing.csv', subset=10, fixed_points=50)
        input_vector, target = ds[0]
        
        # Check input size: 50x + 50y + 1AoA + 1Re = 102
        self.assertEqual(input_vector.shape[0], 102, "Input vector should have 102 features")
        
    def test_scaling_physics(self):
        """Test if Reynolds number is log-scaled and normalized."""
        ds = AirfoilDataset('DeepLearWing.csv', subset=10, fixed_points=50)
        input_vector, _ = ds[0]
        
        # Reynolds is at index 101. 
        # In dataset.py, we did: log10(Re) / 6.0
        # So for Re=200,000 -> log10 is ~5.3 -> divided by 6.0 is ~0.88
        reynolds_norm = input_vector[-1].item()
        
        # We expect a value roughly between 0.6 (Re=4000) and 1.2 (Re=15M)
        self.assertTrue(0.5 < reynolds_norm < 1.5, f"Reynolds value {reynolds_norm} is out of expected normalized range!")

if __name__ == '__main__':
    unittest.main()