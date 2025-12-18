import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import AirfoilDataset
from model import AirfoilMLP
import time
import pandas as pd

# --- Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
SUBSET_SIZE = None  # Set to an integer (e.g., 10000) for debugging

def train():
    """
    Executes the training pipeline:
    1. Loads and splits data.
    2. Initializes the MLP model.
    3. Trains using Adam optimizer and ReduceLROnPlateau scheduler.
    4. Logs metrics to CSV and saves the final model state.
    """
    print("--- INITIALIZING TRAINING PIPELINE ---")
    
    # Auto-detect hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 1. Data Preparation
    full_dataset = AirfoilDataset('DeepLearWing.csv', subset=SUBSET_SIZE, fixed_points=50)
    
    # Split: 70% Train, 15% Val, 15% Test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    print(f"Data Split -> Train: {len(train_data)} | Val: {len(val_data)}")

    # 2. Model Setup
    model = AirfoilMLP(input_size=102).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler: Reduces LR by 50% if validation loss stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    print("\n--- STARTING TRAINING ---")
    start_time = time.time()
    history = []

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train() 
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 4. Validation Step
        model.eval() 
        val_loss = 0.0
        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate Averages
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log Metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr
        })
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Step Scheduler
        scheduler.step(avg_val_loss)

    print("-" * 30)
    print(f"Training Complete in {time.time() - start_time:.2f} seconds.")
    
    # 5. Save Artifacts
    torch.save(model.state_dict(), "airfoil_model_smart.pth")
    print("Model saved as 'airfoil_model_smart.pth'")
    
    df = pd.DataFrame(history)
    df.to_csv("training_log.csv", index=False)
    print("Training Log saved as 'training_log.csv'")

if __name__ == "__main__":
    train()