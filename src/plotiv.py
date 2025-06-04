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
    # Plot 4: Second derivative of gd with respect to Vd (d²gd/dVd²) (bottom right)
    # Plot 4: Second derivative of gd with respect to Vd (d²gd/dVd²) (bottom right)
    ax = axes[1, 1]

    # Use a very different approach - directly work with the first derivative (conductance)
    # and calculate second derivatives with minimal smoothing and scaling

    # Generate a very high resolution grid for evaluation
    vd_min = 0.002  # 2mV
    vd_max = 0.8    # 800mV (or max(unique_vd) if you prefer)

    print("=== DEBUGGING SECOND DERIVATIVE PLOT ===")
    print(f"Data statistics: min Vd = {min(unique_vd)}, max Vd = {max(unique_vd)}")
    print(f"Number of Vd points: {len(unique_vd)}")

    # Try a completely different visualization approach - focus on the derivatives themselves
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Get directly available gd values
        gd_values = gd_grid[:, vg_idx]
        
        # Print min/max stats to understand the scale
        print(f"Vg={vg_val}V: gd min={np.min(gd_values):.6f}, max={np.max(gd_values):.6f}")
        
        # Create a smooth interpolation of gd values 
        # Use cubic spline interpolation for smoother derivatives
        from scipy.interpolate import CubicSpline
        
        # Filter to valid Vd range
        valid_idx = (unique_vd >= vd_min)
        valid_vd = unique_vd[valid_idx]
        valid_gd = gd_values[valid_idx]
        
        try:
            # Create cubic spline of gd vs vd
            cs = CubicSpline(valid_vd, valid_gd)
            
            # Generate a dense evaluation grid
            vd_dense = np.linspace(vd_min, vd_max, 500)
            
            # Evaluate first derivative (dgd/dvd) and second derivative (d²gd/dvd²)
            # directly from the spline
            dgd_dvd = cs.derivative(1)(vd_dense)  # First derivative of gd
            d2gd_dvd2 = cs.derivative(2)(vd_dense)  # Second derivative of gd
            
            # Apply magnification factor if needed - try different values
            magnification = 50  # Adjust as needed to see features
            
            # Plot with magnification
            ax.plot(vd_dense, d2gd_dvd2 * magnification, linestyles[i], color=colors[i], 
                    linewidth=2.0, label=f"{labels[i]} (×{magnification})")
            
            # Print stats to understand scaling
            print(f"Vg={vg_val}V: d2gd/dvd2 min={np.min(d2gd_dvd2):.6f}, max={np.max(d2gd_dvd2):.6f}")
            
        except Exception as e:
            print(f"CubicSpline failed for Vg={vg_val}V: {e}")
            
            # Fall back to direct numerical differentiation with minimal smoothing
            # and try a different scaling
            
            # Apply minimal smoothing to gd values
            gd_smooth = gaussian_filter1d(valid_gd, sigma=0.5)
            
            # Calculate first derivative numerically
            dgd_dvd = np.gradient(gd_smooth, valid_vd)
            dgd_smooth = gaussian_filter1d(dgd_dvd, sigma=0.5)
            
            # Calculate second derivative numerically
            d2gd_dvd2 = np.gradient(dgd_smooth, valid_vd)
            
            # Apply stronger magnification
            magnification = 100  # Try even larger value
            
            # Plot with magnification
            ax.plot(valid_vd, d2gd_dvd2 * magnification, linestyles[i], color=colors[i], 
                    linewidth=2.0, label=f"{labels[i]} (×{magnification})")
            
            # Print stats
            print(f"Fallback method - Vg={vg_val}V: d2gd/dvd2 min={np.min(d2gd_dvd2):.6f}, max={np.max(d2gd_dvd2):.6f}")

    # Try one more completely different method - direct polynomial fitting
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Get id values directly
        id_values = id_grid[:, vg_idx]
        
        # Filter to valid range
        valid_idx = (unique_vd >= vd_min) & (unique_vd <= 0.3)  # Focus on lower Vd range
        valid_vd = unique_vd[valid_idx]
        valid_id = id_values[valid_idx]
        
        try:
            # Use polynomial fitting - can directly get derivatives
            # Try different polynomial degrees
            from numpy.polynomial import Polynomial
            
            # Use a polynomial of suitable degree
            deg = 10  # Higher degree can fit more features but may overfit
            p = Polynomial.fit(valid_vd, valid_id, deg)
            
            # Get the derivatives
            p_d1 = p.deriv(1)  # First derivative (similar to gd)
            p_d2 = p.deriv(2)  # Second derivative (similar to dgd_dvd)
            p_d3 = p.deriv(3)  # Third derivative (similar to d2gd_dvd2)
            
            # Evaluate on a dense grid
            vd_dense = np.linspace(vd_min, 0.3, 300)
            d2gd_poly = p_d3(vd_dense)
            
            # Apply less magnification for polynomial method
            poly_mag = 5
            
            # Plot with dashed lines to distinguish from spline method
            ax.plot(vd_dense, d2gd_poly * poly_mag, '--', color=colors[i], 
                    linewidth=1.0, alpha=0.6)
            
            # Add scatter plot for a few points
            scatter_idx = np.linspace(0, len(vd_dense)-1, 15, dtype=int)
            ax.scatter(vd_dense[scatter_idx], d2gd_poly[scatter_idx] * poly_mag, 
                    color=colors[i], s=20, marker='x')
            
        except Exception as e:
            print(f"Polynomial method failed for Vg={vg_val}V: {e}")

    # Set up the plot with clear visualization
    ax.set_xlabel('$V_{DS}$ (V)')
    ax.set_ylabel('$d^2g_d/dV_{DS}^2$ (scaled)')
    ax.set_xlim(vd_min, vd_max)

    # Auto-adjust y-limits based on visible data
    # Exclude extreme outliers by using percentiles
    all_values = []
    for line in ax.get_lines():
        ydata = line.get_ydata()
        all_values.extend(ydata[~np.isnan(ydata)])

    if all_values:
        low_percentile = np.percentile(all_values, 5)
        high_percentile = np.percentile(all_values, 95)
        
        # Set y-limits with some padding
        y_range = high_percentile - low_percentile
        y_min = low_percentile - 0.1 * y_range
        y_max = high_percentile + 0.1 * y_range
        
        # Make sure we include zero
        y_min = min(y_min, -0.1 * y_max)
        
        ax.set_ylim(y_min, y_max)
    else:
        # Fallback to default limits
        ax.set_ylim(-3, 3)

    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.legend(fontsize=8)  # Smaller font to fit all labels
    ax.set_title('d²gd/dVd²-Vd Characteristics (Enhanced)')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'id_gd_dgd_d2gd_vd_characteristics.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'id_gd_dgd_d2gd_vd_characteristics.pdf'))
    plt.close(fig)

    # Improved version of the second derivative plot with professional styling
    ax = axes[1, 1]

    # Use a clean, consistent approach based on cubic spline interpolation
    vd_min = 0.002  # 2mV
    vd_max = 0.8    # 800mV (or max(unique_vd))

    # Colors and markers for different Vgs values
    colors = ['black', 'red', 'blue', 'green']
    linestyles = ['-', '-', '-', '-']
    markers = ['o', 's', '^', 'd']
    labels = ['Vgs=10mV', 'Vgs=200mV', 'Vgs=400mV', 'Vgs=600mV']

    # Optimal magnification factor based on our previous exploration
    magnification = 50

    # Create high-quality professional plot
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Get conductance values
        gd_values = gd_grid[:, vg_idx]
        
        # Filter to valid Vd range and remove any NaN values
        valid_idx = (unique_vd >= vd_min) & ~np.isnan(gd_values)
        valid_vd = unique_vd[valid_idx]
        valid_gd = gd_values[valid_idx]
        
        # Sort the data points by Vd to ensure proper interpolation
        sort_idx = np.argsort(valid_vd)
        valid_vd = valid_vd[sort_idx]
        valid_gd = valid_gd[sort_idx]
        
        # Use cubic spline interpolation for smooth derivatives
        from scipy.interpolate import CubicSpline
        
        try:
            # Create cubic spline with appropriate boundary conditions
            cs = CubicSpline(valid_vd, valid_gd, bc_type='natural')
            
            # Generate a dense evaluation grid with focus on low Vds region
            # Use non-uniform grid with more points at low Vds
            vd_dense1 = np.linspace(vd_min, 0.1, 200)  # More points in 2-100mV region
            vd_dense2 = np.linspace(0.1, vd_max, 200)  # Fewer points in higher Vds
            vd_dense = np.unique(np.concatenate([vd_dense1, vd_dense2]))
            
            # Evaluate second derivative directly from the spline
            d2gd_dvd2 = cs.derivative(2)(vd_dense)
            
            # Apply light smoothing to reduce any remaining noise
            d2gd_dvd2_smooth = gaussian_filter1d(d2gd_dvd2, sigma=0.5)
            
            # Plot main curve with professional styling
            line, = ax.plot(vd_dense, d2gd_dvd2_smooth * magnification, 
                    linestyle=linestyles[i], color=colors[i], linewidth=1.8, 
                    label=f"{labels[i]}")
            
            # Add selective markers for clarity (not too many)
            # Use logarithmic spacing for marker positions to focus on the transition region
            log_indices = np.unique(np.logspace(0, np.log10(len(vd_dense)-1), 12, dtype=int))
            ax.scatter(vd_dense[log_indices], d2gd_dvd2_smooth[log_indices] * magnification, 
                    color=colors[i], s=35, marker=markers[i], alpha=0.8, zorder=10)
            
        except Exception as e:
            print(f"CubicSpline failed for Vg={vg_val}V: {e}")
            # Fall back to direct numerical differentiation if needed

    # Professional formatting for the plot
    ax.set_xlabel('$V_{DS}$ (V)', fontsize=11)
    ax.set_ylabel('$d^2g_d/dV_{DS}^2$ (A/V$^3$)', fontsize=11)
    ax.set_xlim(vd_min, vd_max)

    # Set y-axis limits to focus on the important features
    # Find appropriate y-limits from the data
    all_ydata = []
    for line in ax.get_lines():
        all_ydata.extend(line.get_ydata())

    if all_ydata:
        # Use percentiles to exclude extreme outliers
        y_min = np.percentile(all_ydata, 1)
        y_max = np.percentile(all_ydata, 99)
        
        # Add some padding and ensure 0 is included
        y_range = y_max - y_min
        y_min = min(y_min - 0.05 * y_range, 0)
        y_max = y_max + 0.05 * y_range
        
        ax.set_ylim(y_min, y_max)

    # Create professional grid and zero line
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)

    # Add a magnification note in the plot
    note_text = f"Note: Values scaled ×{magnification}"
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.3'))

    # Add scientific notation for the y-axis if needed
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Create a neater legend with smaller font
    leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    leg.get_frame().set_edgecolor('gray')

    # Professional title
    ax.set_title('$d^2g_d/dV_{DS}^2$-$V_d$ Characteristics', fontsize=12)

    # Set custom tick spacing for better readability
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    # Create an alternative log-scale plot as a separate figure
    plt.figure(figsize=(7, 5))
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        gd_values = gd_grid[:, vg_idx]
        
        valid_idx = (unique_vd >= vd_min) & ~np.isnan(gd_values)
        valid_vd = unique_vd[valid_idx]
        valid_gd = gd_values[valid_idx]
        
        sort_idx = np.argsort(valid_vd)
        valid_vd = valid_vd[sort_idx]
        valid_gd = valid_gd[sort_idx]
        
        try:
            cs = CubicSpline(valid_vd, valid_gd, bc_type='natural')
            vd_dense = np.logspace(np.log10(vd_min), np.log10(vd_max), 400)
            d2gd_dvd2 = cs.derivative(2)(vd_dense)
            d2gd_dvd2_smooth = gaussian_filter1d(d2gd_dvd2, sigma=0.5)
            
            plt.plot(vd_dense, d2gd_dvd2_smooth * magnification, 
                    linestyle=linestyles[i], color=colors[i], linewidth=1.8, 
                    label=f"{labels[i]}")
            
            log_indices = np.unique(np.geomspace(1, len(vd_dense)-1, 10, dtype=int))
            plt.scatter(vd_dense[log_indices], d2gd_dvd2_smooth[log_indices] * magnification, 
                    color=colors[i], s=35, marker=markers[i], alpha=0.8)
            
        except Exception as e:
            print(f"Log-scale plot failed for Vg={vg_val}V: {e}")

    # Format the log-scale plot
    plt.xscale('log')
    plt.xlabel('$V_{DS}$ (V)', fontsize=11)
    plt.ylabel('$d^2g_d/dV_{DS}^2$ (A/V$^3$)', fontsize=11)
    plt.xlim(vd_min, vd_max)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)

    plt.text(0.98, 0.02, f"Note: Values scaled ×{magnification}", transform=plt.gca().transAxes, 
            fontsize=9, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.3'))

    leg = plt.legend(fontsize=9, loc='upper right', framealpha=0.9)
    leg.get_frame().set_edgecolor('gray')
    plt.title('$d^2g_d/dV_{DS}^2$-$V_d$ Characteristics (Log Scale)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'log_scale_d2gd_dvd2_enhanced.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'log_scale_d2gd_dvd2_enhanced.pdf'))
    plt.close()
    
    # Create a separate figure just for the second derivative with enhanced visibility
    plt.figure(figsize=(8, 6))
    
    # Calculate proper scaling factor to get units in A/V³ (not mA/V³)
    # If your data is in A, then set scaling_factor to 1
    # If your data is in mA (which it seems to be), then we need to convert mA to A (1/1000)
    scaling_factor = 1  # Since we want units in A/V³, not mA/V³
    
    # Skip the very low Vds values to avoid the division-by-zero-like issues
    for i, vg_val in enumerate(actual_vg_values):
        # Generate model predictions but skip the problematic very low Vds values
        vd_model = np.linspace(0.01, max(unique_vd), 300)  # Start at 0.01V instead of nearly 0
        vg_model = np.ones_like(vd_model) * vg_val
        x_model = np.column_stack((vg_model, vd_model))
        x_model_scaled = x_scaler.transform(x_model)
        
        # Get model predictions
        y_model = model.predict(x_model_scaled).numpy().flatten()
        id_model = vd_model * np.exp(y_model)
        
        # Apply stronger smoothing to the original data before differentiation
        id_model_smooth = gaussian_filter1d(id_model, sigma=1.0)
        
        # Calculate derivatives with progressively more smoothing
        gd_model = np.gradient(id_model_smooth, vd_model)
        gd_model_smooth = gaussian_filter1d(gd_model, sigma=1.5)
        
        dgd_model = np.gradient(gd_model_smooth, vd_model)
        dgd_model_smooth = gaussian_filter1d(dgd_model, sigma=2.0)
        
        d2gd_dvd2_model = np.gradient(dgd_model_smooth, vd_model)
        
        # Apply stronger final smoothing
        d2gd_dvd2_model = gaussian_filter1d(d2gd_dvd2_model, sigma=2.5)
        
        # Plot with markers for better visibility (use fewer markers)
        plt.plot(vd_model, d2gd_dvd2_model, linestyles[i], color=colors[i], 
                linewidth=2.0, label=labels[i], marker=markers[i], markevery=30)
    
    # For the actual data points, also skip very low Vds values and apply stronger smoothing
    for i, vg_val in enumerate(actual_vg_values):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        
        # Filter out Vds values below a threshold
        vd_threshold = 0.01  # Skip points below 0.01V
        valid_indices = unique_vd >= vd_threshold
        valid_vd = unique_vd[valid_indices]
        
        # Compute derivatives only for valid Vds values
        gd_values = gd_grid[valid_indices, vg_idx]
        
        # Apply strong smoothing before differentiation
        gd_values_smooth = gaussian_filter1d(gd_values, sigma=2.0)
        
        # Calculate derivatives with careful numerical methods
        dgd_dvd_values = np.gradient(gd_values_smooth, valid_vd)
        dgd_dvd_values_smooth = gaussian_filter1d(dgd_dvd_values, sigma=2.5)
        
        d2gd_dvd2_values = np.gradient(dgd_dvd_values_smooth, valid_vd)
        d2gd_dvd2_values = gaussian_filter1d(d2gd_dvd2_values, sigma=3.0)
        
        # Plot actual data as markers with less frequency
        plt.scatter(valid_vd, d2gd_dvd2_values, color=colors[i], s=40, alpha=0.6, marker=markers[i])
    
    # Match the paper's axis settings exactly
    plt.xlabel('$V_{DS}$ (V)')
    plt.ylabel('$d^2g_d/dV_{DS}^2$ (A/V$^3$)')
    plt.xlim(0.01, 0.8)  # Start from 0.01V to avoid the problematic region
    
    # Set fixed y-axis limits to exactly match the paper
    plt.ylim(-3, 20)  # Match paper axis limits exactly
    
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    plt.title('d²gd/dVd²-Vd Characteristics')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'paper_style_d2gd_dvd2_characteristics.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'paper_style_d2gd_dvd2_characteristics.pdf'))
    plt.close()

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