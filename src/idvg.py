import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_model_and_data(model_path, data_path):
    """Load the model and data file"""
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Sort data for better visualization
    df = df.sort_values(by=['Vg', 'Vd'])
    
    return model, df

def predict_current(model, vg, vd, x_scaler=None):
    """Predict drain current using the loaded model"""
    # Prepare input features
    X = np.column_stack((vg, vd))
    
    # Scale input if scaler is provided
    if x_scaler is not None:
        X = x_scaler.transform(X)
    
    # Get the predicted ln(Id/Vd) values
    y_pred = model.predict(X, verbose=0)
    
    # Convert back to current: Id = Vd * exp(y)
    id_pred = vd * np.exp(y_pred.flatten())
    
    return id_pred

def create_id_vg_plot(model, original_data, output_file='id_vg_multiple_vd.png'):
    """Create an Id-Vg plot showing multiple Vd curves with continuous lines"""
    plt.figure(figsize=(10, 8))
    
    # Get unique Vg and Vd values from original data
    unique_vg = np.sort(np.unique(original_data['Vg'].values))
    unique_vd = np.sort(np.unique(original_data['Vd'].values))
    
    # Specify Vd values to plot (similar to your reference image)
    # Adjust these values based on your data
    vd_values = [0.01, 0.27, 0.55, 0.8]
    
    # Choose colors for different Vd values
    colors = ['blue', 'green', 'red', 'purple']
    
    # Generate smooth Vg values for model predictions
    vg_smooth = np.linspace(np.min(unique_vg), np.max(unique_vg), 200)
    
    # Plot for each Vd value
    for i, vd in enumerate(vd_values):
        # Generate model prediction
        vd_smooth = np.ones_like(vg_smooth) * vd
        id_pred = predict_current(model, vg_smooth, vd_smooth)
        
        # Plot model line
        plt.plot(vg_smooth, id_pred, '-', color=colors[i], linewidth=2.5, 
                 label=f'Vd={vd:.2f}V')
        
        # Plot original data points
        mask = np.isclose(original_data['Vd'].values, vd, atol=1e-2)
        orig_vg = original_data['Vg'].values[mask]
        orig_id = original_data['Id'].values[mask]
        
        # Sort by Vg
        if len(orig_vg) > 0:
            sort_idx = np.argsort(orig_vg)
            orig_vg = orig_vg[sort_idx]
            orig_id = orig_id[sort_idx]
            
            # Plot data points
            plt.plot(orig_vg, orig_id, 'o', color=colors[i], markersize=5)
    
    plt.title('Id-Vg Characteristics (Linear Scale)')
    plt.xlabel('Gate Voltage (V)')
    plt.ylabel('Drain Current (A)')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Id-Vg plot saved to {output_file}")

if __name__ == "__main__":
    # Define paths
    model_path = 'models/iv_model_exact_loss_Full_final.keras'
    data_path = 'iv_data.csv'
    output_file = 'id_vg_multiple_vd.png'
    
    # Load model and data
    model, data = load_model_and_data(model_path, data_path)
    
    # Create plot
    create_id_vg_plot(model, data, output_file)