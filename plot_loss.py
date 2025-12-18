import pandas as pd
import matplotlib.pyplot as plt

def plot_training_history():
    """
    Reads the 'training_log.csv' generated during training and plots 
    the Loss Convergence Curve (MSE vs Epochs).
    """
    try:
        df = pd.read_csv("training_log.csv")
    except FileNotFoundError:
        print("❌ Error: 'training_log.csv' not found. Please run train.py first.")
        return

    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue', marker='o')
    
    # Plot Validation Loss
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', marker='o', linestyle='--')
    
    # Styling
    plt.title("Model Convergence: Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.savefig("loss_curve.png")
    print("Convergence plot saved as 'loss_curve.png'")

if __name__ == "__main__":
    plot_training_history()