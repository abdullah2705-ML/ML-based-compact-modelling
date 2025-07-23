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
from scipy.signal import savgol_filter


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
    vg_values = [0.35, 0.25, 0.3, 0.4] # Vg values for Id-Vd plots (removed 0.80V)
    
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
# Plot Id-Vd, gd-Vd, and dgd_dvd-Vd curves - REFERENCE STYLE
# Plot Id-Vd, gd-Vd, and dgd_dvd-Vd curves - REFERENCE STYLE
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
    
    # Create figure - 2x2 layout to include second derivative
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Colors for different VG values
    colors = ['black', 'red', 'blue', 'green']
    markers = ['o', 'o', 'o', 'o']
    linestyles = ['-', '-', '-', '-']
    labels = ['Vgs=10mV', 'Vgs=200mV', 'Vgs=400mV', 'Vgs=600mV']
    
    # Plot 1: Id-Vd curves (top left)
    ax = axes[0, 0]
    
    # First plot all model predictions to ensure they appear as continuous lines
    for i, vg_val in enumerate(actual_vg_values):
        # Generate model predictions with high resolution for smooth curves
        vd_model = np.linspace(min(unique_vd), max(unique_vd), 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Plot model predictions as solid smooth lines (FIRST, before markers)
        ax.plot(vd_model, id_model * 1000, linestyles[i], color=colors[i], linewidth=1.5, 
               label=labels[i])
    
    # Then plot actual data as markers on top
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Actual data
        id_values = id_grid[:, vg_idx]
        
        # Plot actual data as markers only (AFTER the model lines)
        ax.scatter(unique_vd, id_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$I_D$ (mA)')
    ax.set_xlim(0, max(unique_vd))
    ax.grid(True)
    ax.legend()
    ax.set_title('Id-Vd Characteristics')
    
    # Plot 2: gd-Vd curves (top right)
    ax = axes[0, 1]
    
    # First plot all model predictions
    for i, vg_val in enumerate(actual_vg_values):
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
               label=labels[i])
    
    # Then plot actual data as markers
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Actual data
        gd_values = gd_grid[:, vg_idx]
        
        # Plot actual data as markers only
        ax.scatter(unique_vd, gd_values * 1000, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$g_d$ (mS)')
    ax.set_xlim(0, max(unique_vd))
    ax.grid(True)
    ax.legend()
    ax.set_title('gd-Vd Characteristics')
    
    # Plot 3: dgd_dvd-Vd curves (bottom left)
    ax = axes[1, 0]
    
    # First plot all model predictions
    for i, vg_val in enumerate(actual_vg_values):
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
        ax.plot(vd_model, dgd_model, linestyles[i], color=colors[i], linewidth=1.5, 
               label=labels[i])
    
    # Then plot actual data as markers
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Actual data
        dgd_dvd_values = dgd_dvd_grid[:, vg_idx]
        
        # Plot actual data as markers only
        ax.scatter(unique_vd, dgd_dvd_values, color=colors[i], s=20, alpha=0.7)
    
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$dg_d/dV_{DS}$ (mA/V²)')
    ax.set_xlim(0, max(unique_vd))
    ax.grid(True)
    ax.legend()
    ax.set_title('dgd/dVd-Vd Characteristics')
    
    # Plot 4: Second derivative of gd with respect to Vd (d²gd/dVd²) (bottom right)
    from scipy.signal import savgol_filter

    ax = axes[1, 1]
    vd_min, vd_max = 0.0, 0.8

    for i, vg in enumerate(actual_vg_values):
        # 1) Actual data
        idxs = (unique_vd >= vd_min) & (unique_vd <= vd_max)
        vd_cut = unique_vd[idxs]
        gd_cut = gd_grid[idxs, np.where(unique_vg == vg)[0][0]]

        # Smoothing parameters
        window_length = 11 if len(vd_cut) >= 11 else 7  # Use a larger window if possible
        polyorder = 3

        if len(vd_cut) >= window_length:
            d2_data = savgol_filter(
                gd_cut,
                window_length=window_length,
                polyorder=polyorder,
                deriv=2,
                delta=vd_cut[1] - vd_cut[0]
            )
            ax.scatter(
                vd_cut,
                d2_data * 1e3,         # scale to mA/V^3
                color=colors[i],
                marker='o',
                s=30,
                alpha=0.7
            )

        # 2) Model prediction (use same vd_cut grid as data)
        vd_m = vd_cut
        vg_m = np.full_like(vd_m, vg)
        x_m = x_scaler.transform(np.column_stack((vg_m, vd_m)))
        y_m = model.predict(x_m).numpy().flatten()
        id_m = vd_m * np.exp(y_m)

        gd_m = np.gradient(id_m, vd_m)
        d2_model = savgol_filter(
            gd_m,
            window_length=window_length,
            polyorder=polyorder,
            deriv=2,
            delta=vd_m[1] - vd_m[0]
        )

        # Optional: Apply a vertical offset if needed (uncomment to use)
        # if len(vd_cut) >= window_length:
        #     offset = np.mean((d2_data - d2_model) * 1e3)
        #     d2_model = d2_model + offset / 1e3

        ax.plot(
            vd_m,
            d2_model * 1e3,        # same scale
            linestyle=linestyles[i],
            color=colors[i],
            linewidth=2.0,
            label=labels[i]
        )

    ax.set(
        xlabel='$V_{DS}$ (V)',
        ylabel=r'$d^2g_d/dV_{DS}^2$ (mA/V$^3$)',
        xlim=(vd_min, vd_max),
        title='d²gₙ/dVd² – model vs data'
    )
    ax.grid(True)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'id_gd_dgd_d2gd_vd.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'id_gd_dgd_d2gd_vd.pdf'))
    plt.close(fig)

# Improved d²gd/dVd² plot function
def plot_paper_style_second_derivative(plot_data, model, save_dir='plots'):
    """Create a paper-style visualization of the second derivative matching the reference"""
    unique_vg = plot_data['unique_vg']
    unique_vd = plot_data['unique_vd']
    id_grid = plot_data['id_grid']
    gd_grid = plot_data['gd_grid']
    actual_vg_values = plot_data['actual_vg_values']
    x_scaler = plot_data['x_scaler']
    
    plt.figure(figsize=(7, 6))
    
    # Colors and markers similar to the reference paper
    colors = ['black', 'red', 'blue', 'green']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '-', '-', '-']
    
    # Use the exact same labels as in your existing code
    labels = ['Vgs=200mV', 'Vgs=250mV', 'Vgs=300mV', 'Vgs=400mV']
    
    # We need to focus on the right VDS range - start from 0.01 instead of 0
    # to avoid numerical issues, but still show the important features
    vd_min = 0.01
    vd_max = 0.8
    
    # First plot the experimental data points
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Filter out very low VDS values
        valid_indices = unique_vd >= vd_min
        valid_vd = unique_vd[valid_indices]
        
        # Get the channel conductance values
        gd_values = gd_grid[valid_indices, vg_idx]
        
        # Apply minimal smoothing before differentiation
        gd_smooth = gaussian_filter1d(gd_values, sigma=0.5)
        
        # Calculate first derivative (dgd/dVd)
        dgd_dvd_values = np.gradient(gd_smooth, valid_vd)
        dgd_smooth = gaussian_filter1d(dgd_dvd_values, sigma=0.7)
        
        # Calculate second derivative (d²gd/dVd²)
        d2gd_dvd2_values = np.gradient(dgd_smooth, valid_vd)
        
        # Apply just enough smoothing to reduce noise while preserving features
        d2gd_dvd2_smooth = gaussian_filter1d(d2gd_dvd2_values, sigma=0.8)
        
        # Try an alternative method (apply smoothing to original Id data and then differentiate)
        id_values = id_grid[valid_indices, vg_idx]
        id_smooth = gaussian_filter1d(id_values, sigma=0.5)
        gd_alt = np.gradient(id_smooth, valid_vd)
        gd_alt_smooth = gaussian_filter1d(gd_alt, sigma=0.6)
        dgd_alt = np.gradient(gd_alt_smooth, valid_vd)
        dgd_alt_smooth = gaussian_filter1d(dgd_alt, sigma=0.7)
        d2gd_alt = np.gradient(dgd_alt_smooth, valid_vd)
        d2gd_alt_smooth = gaussian_filter1d(d2gd_alt, sigma=0.8)
        
        # Plot experimental data
        plt.scatter(valid_vd, d2gd_alt_smooth, color=colors[i], s=30, alpha=0.7, marker=markers[i])
    
    # Then plot model predictions as lines
    for i, vg_val in enumerate(actual_vg_values):
        # Generate high-resolution model predictions
        vd_model = np.linspace(vd_min, vd_max, 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Apply minimal smoothing to preserve features
        id_model_smooth = gaussian_filter1d(id_model, sigma=0.5)
        
        # Calculate first derivative (gd)
        gd_model = np.gradient(id_model_smooth, vd_model)
        gd_model_smooth = gaussian_filter1d(gd_model, sigma=0.6)
        
        # Calculate second derivative (dgd/dVd)
        dgd_model = np.gradient(gd_model_smooth, vd_model)
        dgd_model_smooth = gaussian_filter1d(dgd_model, sigma=0.7)
        
        # Calculate third derivative (d²gd/dVd²)
        d2gd_model = np.gradient(dgd_model_smooth, vd_model)
        d2gd_model_smooth = gaussian_filter1d(d2gd_model, sigma=0.8)
        
        # Plot model as continuous line
        plt.plot(vd_model, d2gd_model_smooth, linestyles[i], color=colors[i], 
                linewidth=1.5, label=labels[i])
    
    # Set up the plot to match the reference paper style
    plt.xlabel('$V_{DS}$ (V)', fontsize=12)
    plt.ylabel('$d^2g_d/dV_{DS}^2$ (A/V$^3$)', fontsize=12)
    
    # Set fixed axis limits to match the reference paper
    plt.xlim(vd_min, vd_max)
    plt.ylim(-3, 3)  # Match the reference paper's scale
    
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('$d^2g_d/dV_{DS}^2$-$V_d$ Characteristics', fontsize=14)
    
    # Use tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'paper_style_d2gd_dvd2.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'paper_style_d2gd_dvd2.pdf'))
    plt.close()

def plot_refined_low_vds_second_derivative(plot_data, model, save_dir='plots'):
    """Refined low VDS plot that avoids numerical instabilities"""
    unique_vg = plot_data['unique_vg']
    unique_vd = plot_data['unique_vd']
    id_grid = plot_data['id_grid']
    actual_vg_values = plot_data['actual_vg_values']
    x_scaler = plot_data['x_scaler']
    
    plt.figure(figsize=(8, 6))
    
    # Colors and markers
    colors = ['black', 'red', 'blue', 'green']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '-', '-', '-']
    labels = ['Vgs=200mV', 'Vgs=250mV', 'Vgs=300mV', 'Vgs=400mV']
    
    # Set a safer minimum VDS to avoid the extreme instability region
    vd_min = 0.002  # 2mV should avoid most of the problematic region
    vd_max = 0.1    # Focus on first 100mV where most features are
    
    # Plot model predictions with high resolution
    for i, vg_val in enumerate(actual_vg_values):
        # Generate model predictions with spaced points in the stable region
        vd_model = np.linspace(vd_min, vd_max, 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Apply moderate smoothing before differentiation
        id_model_smooth = gaussian_filter1d(id_model, sigma=0.8)
        
        # Calculate derivatives with careful smoothing
        # First derivative (gd)
        gd_model = np.gradient(id_model_smooth, vd_model)
        gd_model_smooth = gaussian_filter1d(gd_model, sigma=1.0)
        
        # Second derivative (dgd/dVd)
        dgd_model = np.gradient(gd_model_smooth, vd_model)
        dgd_model_smooth = gaussian_filter1d(dgd_model, sigma=1.2)
        
        # Third derivative (d²gd/dVd²)
        d2gd_model = np.gradient(dgd_model_smooth, vd_model)
        d2gd_model_smooth = gaussian_filter1d(d2gd_model, sigma=1.5)
        
        # Plot lines
        plt.plot(vd_model, d2gd_model_smooth, linestyles[i], color=colors[i], 
                linewidth=1.5, label=labels[i])
    
    # Now add data points from actual measurements, but skip the unstable region
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Filter to the stable region only
        stable_indices = (unique_vd >= vd_min) & (unique_vd <= vd_max)
        stable_vds = unique_vd[stable_indices]
        
        # Only process if we have enough points
        if len(stable_vds) > 5:
            # Get the current values in the stable region
            id_values = id_grid[stable_indices, vg_idx]
            
            # Apply proper smoothing before differentiation
            id_smooth = gaussian_filter1d(id_values, sigma=0.8)
            
            # Calculate derivatives with careful smoothing
            gd_values = np.gradient(id_smooth, stable_vds)
            gd_smooth = gaussian_filter1d(gd_values, sigma=1.0)
            
            dgd_values = np.gradient(gd_smooth, stable_vds)
            dgd_smooth = gaussian_filter1d(dgd_values, sigma=1.2)
            
            d2gd_values = np.gradient(dgd_smooth, stable_vds)
            d2gd_smooth = gaussian_filter1d(d2gd_values, sigma=1.5)
            
            # Plot data points with reduced marker size and every nth point
            # to avoid overcrowding
            n = max(1, len(stable_vds) // 20)  # Show about 20 markers
            plt.scatter(stable_vds[::n], d2gd_smooth[::n], color=colors[i], 
                       s=30, alpha=0.7, marker=markers[i])
    
    # Set up the plot
    plt.xlabel('$V_{DS}$ (V)', fontsize=12)
    plt.ylabel('$d^2g_d/dV_{DS}^2$ (A/V$^3$)', fontsize=12)
    
    # Use linear scale for x-axis in this plot
    plt.xlim(vd_min, vd_max)
    plt.ylim(-3, 3)  # Constrain y-axis to match reference
    
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Refined Low-$V_{DS}$ $d^2g_d/dV_{DS}^2$-$V_d$ Characteristics', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'refined_low_vds_d2gd_dvd2.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'refined_low_vds_d2gd_dvd2.pdf'))
    plt.close()
    
    # Create a second plot with log scale for x-axis
    plt.figure(figsize=(8, 6))
    
    # Replot the same data with log x-scale
    for i, vg_val in enumerate(actual_vg_values):
        # Generate model predictions
        vd_model = np.linspace(vd_min, vd_max, 300)
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions and derivatives (same as above)
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        id_model_smooth = gaussian_filter1d(id_model, sigma=0.8)
        gd_model = np.gradient(id_model_smooth, vd_model)
        gd_model_smooth = gaussian_filter1d(gd_model, sigma=1.0)
        dgd_model = np.gradient(gd_model_smooth, vd_model)
        dgd_model_smooth = gaussian_filter1d(dgd_model, sigma=1.2)
        d2gd_model = np.gradient(dgd_model_smooth, vd_model)
        d2gd_model_smooth = gaussian_filter1d(d2gd_model, sigma=1.5)
        
        # Plot lines
        plt.plot(vd_model, d2gd_model_smooth, linestyles[i], color=colors[i], 
                linewidth=1.5, label=labels[i])
    
    # Set up the log-scale plot
    plt.xlabel('$V_{DS}$ (V)', fontsize=12)
    plt.ylabel('$d^2g_d/dV_{DS}^2$ (A/V$^3$)', fontsize=12)
    
    # Use log scale for x-axis
    plt.xscale('log')
    plt.xlim(vd_min, vd_max)
    plt.ylim(-3, 3)
    
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Log-Scale Low-$V_{DS}$ $d^2g_d/dV_{DS}^2$-$V_d$ Characteristics', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'log_scale_low_vds_d2gd_dvd2.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'log_scale_low_vds_d2gd_dvd2.pdf'))
    plt.close()

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

         # Call the new improved second derivative plot function
        # print("Generating improved second derivative plot...")
        # plot_improved_second_derivative(plot_data, model)

        # In your main() function, add:
        print("Generating paper-style second derivative plot...")
        plot_paper_style_second_derivative(plot_data, model)

        # print("Generating low-Vds second derivative plot...")
        # plot_low_vds_second_derivative(plot_data, model)
        # In your main() function, add:
        print("Generating refined low-Vds second derivative plot...")
        plot_refined_low_vds_second_derivative(plot_data, model)
        print("Plotting completed. Check the 'plots' directory for the generated images.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()