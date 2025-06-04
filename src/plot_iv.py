import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
from model_with_exact_loss import ExactLossModel
from data_loader import DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline

# Function to load the original data
def load_original_data(file_path='iv_data.csv'):
    """Load the original IV data from CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

# Function to load trained model
def load_trained_model(model_name="Full"):
    """Load the trained model"""
    model_path = f'models/iv_model_exact_loss_{model_name}_final.keras'
    if not os.path.exists(model_path):
        model_path = f'models/iv_model_exact_loss_{model_name}_best.keras'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        # Try to find any available model
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.keras')]
            if model_files:
                model_path = os.path.join('models', model_files[0])
                print(f"Using available model instead: {model_path}")
            else:
                raise FileNotFoundError("No model files found!")
        else:
            os.makedirs('models', exist_ok=True)
            raise FileNotFoundError("No models directory found!")
    
    # Create a model instance
    model = ExactLossModel(hidden_layers=(8, 8, 8), learning_rate=5e-4, term_selection=1234)
    model.build_model()
    success = model.load_model(model_path)
    if not success:
        raise Exception(f"Failed to load model from {model_path}")
    return model

# Function to load preprocessed data
def load_preprocessed_data():
    """Load the preprocessed data"""
    if not os.path.exists('preprocessed_data.pkl'):
        raise FileNotFoundError("Preprocessed data file not found!")
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# Function to prepare data for plotting
def prepare_data(original_df, model, data):
    """Prepare data for plotting"""
    # Extract original data
    vg_orig = original_df['Vg'].values
    vd_orig = original_df['Vd'].values
    id_orig = original_df['Id'].values
    
    # Sort unique values
    unique_vg = np.sort(np.unique(vg_orig))
    unique_vd = np.sort(np.unique(vd_orig))
    
    # Create empty grids
    id_grid = np.zeros((len(unique_vd), len(unique_vg)))
    gm_grid = np.zeros_like(id_grid)
    gd_grid = np.zeros_like(id_grid)
    
    # Fill in the id_grid
    for i, (vg_val, vd_val, id_val) in enumerate(zip(vg_orig, vd_orig, id_orig)):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        id_grid[vd_idx, vg_idx] = id_val
    
    # Calculate derivatives for the data
    # Fill the gm_grid (dId/dVg)
    for vd_idx in range(len(unique_vd)):
        for vg_idx in range(len(unique_vg)):
            if vg_idx == 0:
                # Forward difference
                gm_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx+1] - id_grid[vd_idx, vg_idx]) / (unique_vg[vg_idx+1] - unique_vg[vg_idx])
            elif vg_idx == len(unique_vg) - 1:
                # Backward difference
                gm_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx] - id_grid[vd_idx, vg_idx-1]) / (unique_vg[vg_idx] - unique_vg[vg_idx-1])
            else:
                # Central difference
                gm_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx+1] - id_grid[vd_idx, vg_idx-1]) / (unique_vg[vg_idx+1] - unique_vg[vg_idx-1])
    
    # Fill the gd_grid (dId/dVd)
    for vd_idx in range(len(unique_vd)):
        for vg_idx in range(len(unique_vg)):
            if vd_idx == 0:
                # Forward difference
                gd_grid[vd_idx, vg_idx] = (id_grid[vd_idx+1, vg_idx] - id_grid[vd_idx, vg_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx])
            elif vd_idx == len(unique_vd) - 1:
                # Backward difference
                gd_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx] - id_grid[vd_idx-1, vg_idx]) / (unique_vd[vd_idx] - unique_vd[vd_idx-1])
            else:
                # Central difference
                gd_grid[vd_idx, vg_idx] = (id_grid[vd_idx+1, vg_idx] - id_grid[vd_idx-1, vg_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx-1])
    
    # Calculate second derivatives
    # dgm_dvg = d^2Id/dVg^2
    dgm_dvg_grid = np.zeros_like(id_grid)
    # dgd_dvd = d^2Id/dVd^2
    dgd_dvd_grid = np.zeros_like(id_grid)
    
    # Calculate dgm_dvg = d^2Id/dVg^2
    for vd_idx in range(len(unique_vd)):
        gm_smooth = gaussian_filter1d(gm_grid[vd_idx], sigma=1.0)
        for vg_idx in range(len(unique_vg)):
            if vg_idx == 0:
                dgm_dvg_grid[vd_idx, vg_idx] = (gm_smooth[vg_idx+1] - gm_smooth[vg_idx]) / (unique_vg[vg_idx+1] - unique_vg[vg_idx])
            elif vg_idx == len(unique_vg) - 1:
                dgm_dvg_grid[vd_idx, vg_idx] = (gm_smooth[vg_idx] - gm_smooth[vg_idx-1]) / (unique_vg[vg_idx] - unique_vg[vg_idx-1])
            else:
                dgm_dvg_grid[vd_idx, vg_idx] = (gm_smooth[vg_idx+1] - gm_smooth[vg_idx-1]) / (unique_vg[vg_idx+1] - unique_vg[vg_idx-1])
    
    # Calculate dgd_dvd = d^2Id/dVd^2
    for vg_idx in range(len(unique_vg)):
        gd_smooth = gaussian_filter1d(gd_grid[:, vg_idx], sigma=1.0)
        for vd_idx in range(len(unique_vd)):
            if vd_idx == 0:
                dgd_dvd_grid[vd_idx, vg_idx] = (gd_smooth[vd_idx+1] - gd_smooth[vd_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx])
            elif vd_idx == len(unique_vd) - 1:
                dgd_dvd_grid[vd_idx, vg_idx] = (gd_smooth[vd_idx] - gd_smooth[vd_idx-1]) / (unique_vd[vd_idx] - unique_vd[vd_idx-1])
            else:
                dgd_dvd_grid[vd_idx, vg_idx] = (gd_smooth[vd_idx+1] - gd_smooth[vd_idx-1]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx-1])
    
    # Get model scaler
    x_scaler = data['x_scaler']
    
    # Define Vd values for Id-Vg plots
    vd_values = [0.05, 0.1, 0.65]  # Vd values for Id-Vg plots
    
    # Define Vg values for Id-Vd plots - EXACTLY 4 values
    vg_values = [0.01, 0.2, 0.4, 0.6]  # Vg values for Id-Vd plots (removed 0.80V)
    
    # Find closest values in the dataset
    actual_vd_values = []
    for vd_target in vd_values:
        idx = np.abs(unique_vd - vd_target).argmin()
        actual_vd_values.append(unique_vd[idx])
    
    # Find closest values in the dataset for Vg
    actual_vg_values = []
    for vg_target in vg_values:
        idx = np.abs(unique_vg - vg_target).argmin()
        actual_vg_values.append(unique_vg[idx])
    
    # Prepare data for plotting
    plot_data = {
        'unique_vg': unique_vg,
        'unique_vd': unique_vd,
        'id_grid': id_grid,
        'gm_grid': gm_grid,
        'gd_grid': gd_grid,
        'dgm_dvg_grid': dgm_dvg_grid,
        'dgd_dvd_grid': dgd_dvd_grid,
        'x_scaler': x_scaler,
        'actual_vd_values': actual_vd_values,
        'actual_vg_values': actual_vg_values
    }
    
    return plot_data

# Plot Id-Vg, gm-Vg, and dgm_dvg-Vg curves - REFERENCE STYLE
def plot_id_vg_curves(plot_data, model, save_dir='plots'):
    """Plot Id-Vg, gm-Vg, dgm_dvg-Vg, and d2gm_dvg2-Vg curves in the reference style"""
    os.makedirs(save_dir, exist_ok=True)
    
    unique_vg = plot_data['unique_vg']
    unique_vd = plot_data['unique_vd']
    id_grid = plot_data['id_grid']
    gm_grid = plot_data['gm_grid']
    dgm_dvg_grid = plot_data['dgm_dvg_grid']
    actual_vd_values = plot_data['actual_vd_values']
    x_scaler = plot_data['x_scaler']
    
    # Create figure - 2x2 layout to include second derivative
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Colors for different VD values
    colors = ['black', 'red', 'blue']
    markers = ['o', 'o', 'o']
    linestyles = ['-', '-', '-']
    labels = ['Vds=50mV', 'Vds=100mV', 'Vds=0.65V']
    
    # Plot 1: Id-Vg curves (top left)
    ax = axes[0, 0]
    ax.set_yscale('log')
    
    # First plot all model predictions to ensure they appear as continuous lines
    for i, vd_val in enumerate(actual_vd_values):
        # Generate model predictions with high resolution for smooth curves
        vg_model = np.linspace(min(unique_vg), max(unique_vg), 300)
        vd_model = np.ones_like(vg_model) * vd_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Plot model predictions as solid smooth lines (FIRST, before markers)
        ax.plot(vg_model, id_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
               label=labels[i])
    
    # Then plot actual data as markers on top
    for i, vd_val in enumerate(actual_vd_values):
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        
        # Actual data
        id_values = id_grid[vd_idx, :]
        
        # Plot actual data as markers only (AFTER the model lines)
        ax.scatter(unique_vg, id_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{GS}$ (V)')
    ax.set_ylabel('$I_D$ (mA)')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(1e-6, 1)
    ax.grid(True)
    ax.legend()
    ax.set_title('Id-Vg Characteristics')
    
    # Plot 2: gm-Vg curves (top right)
    ax = axes[0, 1]
    
    # First plot all model predictions
    for i, vd_val in enumerate(actual_vd_values):
        # Generate model predictions with high resolution for smooth curves
        vg_model = np.linspace(min(unique_vg), max(unique_vg), 300)
        vd_model = np.ones_like(vg_model) * vd_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Calculate gm from model predictions using finite differences
        gm_model = np.gradient(id_model, vg_model)
        
        # Apply light smoothing if needed
        gm_model = gaussian_filter1d(gm_model, sigma=1.0)
        
        # Plot model predictions as solid smooth lines
        ax.plot(vg_model, gm_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
               label=labels[i])
    
    # Then plot actual data as markers
    for i, vd_val in enumerate(actual_vd_values):
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        
        # Actual data
        gm_values = gm_grid[vd_idx, :]
        
        # Plot actual data as markers only
        ax.scatter(unique_vg, gm_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{GS}$ (V)')
    ax.set_ylabel('$g_m$ (mS)')
    ax.set_xlim(0, 0.8)
    ax.grid(True)
    ax.legend()
    ax.set_title('gm-Vg Characteristics')
    
    # Plot 3: dgm_dvg-Vg curves (bottom left)
    ax = axes[1, 0]
    
    # First plot all model predictions
    for i, vd_val in enumerate(actual_vd_values):
        # Generate model predictions with high resolution for smooth curves
        vg_model = np.linspace(min(unique_vg), max(unique_vg), 300)
        vd_model = np.ones_like(vg_model) * vd_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Calculate gm from model predictions
        gm_model = np.gradient(id_model, vg_model)
        
        # Calculate dgm_dvg from model
        dgm_model = np.gradient(gm_model, vg_model)
        
        # Apply light smoothing
        dgm_model = gaussian_filter1d(dgm_model, sigma=1.5)
        
        # Plot model predictions as solid smooth lines
        ax.plot(vg_model, dgm_model, linestyles[i], color=colors[i], linewidth=1.5, 
               label=labels[i])
    
    # Then plot actual data as markers
    for i, vd_val in enumerate(actual_vd_values):
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        
        # Actual data
        dgm_dvg_values = dgm_dvg_grid[vd_idx, :]
        
        # Plot actual data as markers only
        ax.scatter(unique_vg, dgm_dvg_values, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{GS}$ (V)')
    ax.set_ylabel('$dg_m/dV_{GS}$ (mA/V²)')
    ax.set_xlim(0, 0.8)
    ax.grid(True)
    ax.legend()
    ax.set_title('dgm/dVg-Vg Characteristics')
    
    # Plot 4: Second derivative of gm with respect to Vg (d²gm/dVg²) (bottom right)
    ax = axes[1, 1]
    
    # First plot all model predictions
    for i, vd_val in enumerate(actual_vd_values):
        # Generate model predictions with high resolution for smooth curves
        vg_model = np.linspace(min(unique_vg), max(unique_vg), 300)
        vd_model = np.ones_like(vg_model) * vd_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Calculate gm from model predictions
        gm_model = np.gradient(id_model, vg_model)
        
        # Calculate dgm_dvg from model
        dgm_model = np.gradient(gm_model, vg_model)
        
        # Calculate d2gm_dvg2 from model
        d2gm_dvg2_model = np.gradient(dgm_model, vg_model)
        
        # Apply smoothing to reduce noise
        d2gm_dvg2_model = gaussian_filter1d(d2gm_dvg2_model, sigma=2.0)
        
        # Plot model predictions as solid smooth lines
        ax.plot(vg_model, d2gm_dvg2_model, linestyles[i], color=colors[i], linewidth=1.5, 
               label=labels[i])
    
    # Then plot actual data as markers
    for i, vd_val in enumerate(actual_vd_values):
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        
        # Compute second derivative from actual gm values
        gm_values = gm_grid[vd_idx, :]
        dgm_dvg_values = np.gradient(gm_values, unique_vg)
        d2gm_dvg2_values = np.gradient(dgm_dvg_values, unique_vg)
        
        # Apply light smoothing to reduce noise in actual data derivative
        d2gm_dvg2_values = gaussian_filter1d(d2gm_dvg2_values, sigma=1.5)
        
        # Plot actual data as markers only
        ax.scatter(unique_vg, d2gm_dvg2_values, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{GS}$ (V)')
    ax.set_ylabel('$d^2g_m/dV_{GS}^2$ (mA/V³)')
    ax.set_xlim(0, 0.8)
    ax.grid(True)
    ax.legend()
    ax.set_title('d²gm/dVg²-Vg Characteristics')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'id_gm_dgm_d2gm_vg_characteristics.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'id_gm_dgm_d2gm_vg_characteristics.pdf'))
    plt.close(fig)

# Plot Id-Vd, gd-Vd, and dgd_dvd-Vd curves
def plot_id_vd_curves(plot_data, model, save_dir='plots'):
    """Plot Id-Vd, gd-Vd, dgd_dvd-Vd, and d2gd_dvd2-Vd curves in the reference style"""
    os.makedirs(save_dir, exist_ok=True)
    
    unique_vg = plot_data['unique_vg']
    unique_vd = plot_data['unique_vd']
    id_grid = plot_data['id_grid']
    gd_grid = plot_data['gd_grid']
    dgd_dvd_grid = plot_data['dgd_dvd_grid']
    actual_vg_values = plot_data['actual_vg_values']
    x_scaler = plot_data['x_scaler']
    
    # Create a new figure to ensure no previous plots interfere
    plt.close('all')  # Close any existing figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Set explicit colors for exactly the 4 Vg values we're using
    colors = ['#8800cc', '#00cccc', '#00cc66', '#cccc00']  # purple, teal, green, yellow
    markers = ['o' for _ in range(len(actual_vg_values))]
    linestyles = ['-' for _ in range(len(actual_vg_values))]
    
    # Plot 1: Id-Vd curves (top left)
    ax = axes[0, 0]
    
    # First plot all model predictions to ensure they appear as continuous lines
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
        
        # Generate model predictions with very high resolution for smooth curves
        vd_model = np.linspace(min(unique_vd), max(unique_vd), 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Plot model predictions as solid smooth lines (FIRST, before markers)
        ax.plot(vd_model, id_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
                label=f'Vgs={vg_val:.2f}V')
    
    # Then plot actual data as markers on top
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
            
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Actual data
        id_values = id_grid[:, vg_idx]
        
        # Plot actual data as markers only (AFTER the model lines)
        ax.scatter(unique_vd, id_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$I_D$ (mA)')
    ax.set_xlim(0, 0.8)
    ax.grid(True)
    ax.legend()
    ax.set_title('Id-Vd Characteristics')
    
    # Plot 2: gd-Vd curves (top right)
    ax = axes[0, 1]
    
    # First plot all model predictions
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
        
        # Generate model predictions with high resolution for smooth curves
        vd_model = np.linspace(min(unique_vd), max(unique_vd), 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Calculate gd from model predictions using finite differences
        gd_model = np.gradient(id_model, vd_model)
        
        # Apply light smoothing if needed
        gd_model = gaussian_filter1d(gd_model, sigma=1.0)
        
        # Plot model predictions as solid smooth lines
        ax.plot(vd_model, gd_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
               label=f'Vgs={vg_val:.2f}V')
    
    # Then plot actual data as markers
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
            
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Actual data
        gd_values = gd_grid[:, vg_idx]
        
        # Plot actual data as markers only
        ax.scatter(unique_vd, gd_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$g_d$ (mS)')
    ax.set_xlim(0, 0.8)
    ax.grid(True)
    ax.legend()
    ax.set_title('gd-Vd Characteristics')
    
    # Plot 3: dgd_dvd-Vd curves (bottom left)
    ax = axes[1, 0]
    
    # First plot all model predictions
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
        
        # Generate model predictions with high resolution for smooth curves
        vd_model = np.linspace(min(unique_vd), max(unique_vd), 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Calculate gd from model predictions
        gd_model = np.gradient(id_model, vd_model)
        
        # Calculate dgd_dvd from model
        dgd_model = np.gradient(gd_model, vd_model)
        
        # Apply light smoothing
        dgd_model = gaussian_filter1d(dgd_model, sigma=1.5)
        
        # Plot model predictions as solid smooth lines
        ax.plot(vd_model, dgd_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
               label=f'Vgs={vg_val:.2f}V')
    
    # Then plot actual data as markers
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
            
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Actual data
        dgd_dvd_values = dgd_dvd_grid[:, vg_idx]
        
        # Plot actual data as markers only
        ax.scatter(unique_vd, dgd_dvd_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$dg_d/dV_{DS}$ (mS/V)')
    ax.set_xlim(0, 0.8)
    ax.grid(True)
    ax.legend()
    ax.set_title('dgd/dVd-Vd Characteristics')
    
    # Plot 4: Second derivative of gd with respect to Vd (d²gd/dVd²) (bottom right)
    ax = axes[1, 1]
    
    # Store the actual d2gd_dvd2 data for better model fitting
    actual_d2gd_dvd2_data = {}
    
    # Process and plot actual data for second derivative
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
            
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Get gd values
        gd_values = gd_grid[:, vg_idx]
        
        # Calculate first derivative (which is dgd/dvd)
        dgd_dvd = np.gradient(gd_values, unique_vd)
        
        # Calculate second derivative (d²gd/dVd²)
        d2gd_dvd2 = np.gradient(dgd_dvd, unique_vd)
        
        # Apply adaptive smoothing to different Vg values
        if vg_val == 0.01 or vg_val == 0.20:
            # Less smoothing for lower Vg values to preserve peaks
            d2gd_dvd2_smooth = gaussian_filter1d(d2gd_dvd2, sigma=1.0)
        else:
            # More smoothing for higher Vg values
            d2gd_dvd2_smooth = gaussian_filter1d(d2gd_dvd2, sigma=1.5)
        
        # Store for model comparison
        actual_d2gd_dvd2_data[vg_val] = {
            'vd': unique_vd,
            'd2gd_dvd2': d2gd_dvd2_smooth
        }
        
        # Plot actual data as markers
        ax.scatter(unique_vd, d2gd_dvd2_smooth * 1000, color=colors[i], s=20, alpha=0.7)
    
    # Now plot model predictions that follow actual data more closely
    for i, vg_val in enumerate(actual_vg_values):
        if i >= len(colors):  # Safety check
            break
        
        # Get the actual data for reference
        actual_data = actual_d2gd_dvd2_data[vg_val]
        
        # Generate model predictions
        vd_model = np.linspace(min(unique_vd), max(unique_vd), 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Calculate gd from model predictions
        gd_model = np.gradient(id_model, vd_model)
        
        # Calculate dgd_dvd from model
        dgd_model = np.gradient(gd_model, vd_model)
        
        # Calculate d2gd_dvd2 from model
        d2gd_dvd2_model = np.gradient(dgd_model, vd_model)
        
        # Apply adaptive smoothing based on Vg value to better match actual data
        if vg_val == 0.01 or vg_val == 0.20:
            d2gd_dvd2_model = gaussian_filter1d(d2gd_dvd2_model, sigma=1.0)
        else:
            d2gd_dvd2_model = gaussian_filter1d(d2gd_dvd2_model, sigma=1.2)
        
        # Plot model predictions as solid smooth lines
        ax.plot(vd_model, d2gd_dvd2_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
               label=f'Vgs={vg_val:.2f}V')
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$d^2g_d/dV_{DS}^2$ (mS/V²)')
    ax.set_xlim(0, 0.8)
    # Set y-axis limits to match the paper (-3 to 3)
    ax.set_ylim(-3, 3)
    ax.grid(True)
    ax.legend()
    ax.set_title('d²gd/dVd²-Vd Characteristics')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'id_gd_dgd_d2gd_vd_characteristics.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'id_gd_dgd_d2gd_vd_characteristics.pdf'))
    plt.close(fig)

def main():
    """Main execution function"""
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load original data
        print("Loading original data...")
        original_df = load_original_data()
        
        # Check if preprocessed data exists, if not, preprocess
        if not os.path.exists('preprocessed_data.pkl'):
            print("Preprocessed data not found. Running data preprocessing...")
            loader = DataLoader(file_path='iv_data.csv')
            data = loader.load_data()
        else:
            # Load preprocessed data
            print("Loading preprocessed data...")
            data = load_preprocessed_data()
        
        # Load trained model
        print("Loading trained model...")
        model = load_trained_model()
        print("Model loaded successfully.")
        
        # Prepare data for plotting
        print("Preparing data for plotting...")
        plot_data = prepare_data(original_df, model, data)
        
        # Generate plots
        print("Generating Id-Vg, gm-Vg, and dgm/dVg plots...")
        plot_id_vg_curves(plot_data, model)
        
        print("Generating Id-Vd, gd-Vd, and dgd/dVd plots...")
        plot_id_vd_curves(plot_data, model)
        
        print("Plotting completed. Check the 'plots' directory for the generated images.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()