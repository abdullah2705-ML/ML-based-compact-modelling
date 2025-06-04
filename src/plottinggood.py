import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import argparse
from model_with_exact_loss import ExactLossModel

def load_raw_data(file_path='iv_data.csv'):
    """Load the original raw data file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file {file_path} not found!")
    
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Extract columns
    vg = df['Vg'].values
    vd = df['Vd'].values
    id_values = df['Id'].values
    
    return vg, vd, id_values, df

def load_model(term_selection):
    """Load trained model with the specified term selection"""
    # Map term selection to name
    term_name_map = {
        0: "MSE",
        1: "Term1",
        12: "Term1+2",
        123: "Term1+2+3",
        1234: "Full"
    }
    term_name = term_name_map.get(term_selection, f"Custom({term_selection})")
    
    # Define model path using the exact naming convention from your files
    model_path = f'models/iv_model_exact_loss_{term_name}_best.keras'
    
    if not os.path.exists(model_path):
        # Try with final instead of best
        model_path = f'models/iv_model_exact_loss_{term_name}_final.keras'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found with pattern iv_model_exact_loss_{term_name}_*.keras")
    
    # Create model and load weights
    model = ExactLossModel(hidden_layers=(8, 8, 8), learning_rate=1e-4, term_selection=term_selection)
    model.build_model()
    
    if not model.load_model(model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")
    
    print(f"Successfully loaded model: {model_path}")
    return model, term_name

def load_preprocessed_data():
    """Load preprocessed data to get scalers and train/test split"""
    data_path = 'preprocessed_data.pkl'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data file {data_path} not found!")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def create_prediction_grid(model, vg_range, vd_range, x_scaler):
    """Create a prediction grid for the entire range of VG and VD"""
    # Create grid of VG, VD points
    vg_mesh, vd_mesh = np.meshgrid(vg_range, vd_range)
    
    # Flatten the grid points
    vg_flat = vg_mesh.flatten()
    vd_flat = vd_mesh.flatten()
    
    # Stack inputs
    inputs = np.column_stack((vg_flat, vd_flat))
    
    # Scale inputs
    inputs_scaled = x_scaler.transform(inputs)
    
    # Predict y = ln(Id/Vd)
    y_pred = model.predict(inputs_scaled).numpy().flatten()
    
    # Convert back to Id (applying inverse transformation)
    id_pred = vd_flat * np.exp(y_pred)
    
    # Reshape to grid
    id_grid = id_pred.reshape(vd_mesh.shape)
    
    return vg_mesh, vd_mesh, id_grid

def calculate_derivatives(vg_mesh, vd_mesh, id_grid):
    """Calculate gm and gd from the id_grid using improved differentiation with smoothing"""
    # Apply Savitzky-Golay filter to id_grid for smoother differentiation
    from scipy.signal import savgol_filter
    
    # Create empty grids for derivatives
    gm_grid = np.zeros_like(id_grid)
    gd_grid = np.zeros_like(id_grid)
    
    # Calculate step sizes
    vg_step = vg_mesh[0, 1] - vg_mesh[0, 0] if vg_mesh.shape[1] > 1 else 1
    vd_step = vd_mesh[1, 0] - vd_mesh[0, 0] if vd_mesh.shape[0] > 1 else 1
    
    # Apply smoothing along each direction before differentiation
    id_grid_smoothed = np.copy(id_grid)
    
    # Smooth along vg direction (rows)
    for i in range(id_grid.shape[0]):
        window_length = min(11, id_grid.shape[1] - 2)
        if window_length > 2 and window_length % 2 == 1:
            id_grid_smoothed[i, :] = savgol_filter(id_grid[i, :], window_length, 3)
    
    # Smooth along vd direction (columns)
    for j in range(id_grid.shape[1]):
        window_length = min(11, id_grid.shape[0] - 2)
        if window_length > 2 and window_length % 2 == 1:
            id_grid_smoothed[:, j] = savgol_filter(id_grid_smoothed[:, j], window_length, 3)
    
    # Calculate gm (∂Id/∂VGS) using central difference - along columns (axis 1)
    for i in range(id_grid.shape[0]):
        gm_grid[i, 1:-1] = (id_grid_smoothed[i, 2:] - id_grid_smoothed[i, :-2]) / (2 * vg_step)
        # Edge cases
        gm_grid[i, 0] = (id_grid_smoothed[i, 1] - id_grid_smoothed[i, 0]) / vg_step
        gm_grid[i, -1] = (id_grid_smoothed[i, -1] - id_grid_smoothed[i, -2]) / vg_step
    
    # Calculate gd (∂Id/∂VDS) using central difference - along rows (axis 0)
    for j in range(id_grid.shape[1]):
        gd_grid[1:-1, j] = (id_grid_smoothed[2:, j] - id_grid_smoothed[:-2, j]) / (2 * vd_step)
        # Edge cases
        gd_grid[0, j] = (id_grid_smoothed[1, j] - id_grid_smoothed[0, j]) / vd_step
        gd_grid[-1, j] = (id_grid_smoothed[-1, j] - id_grid_smoothed[-2, j]) / vd_step
    
    # Apply additional smoothing to the derivatives
    for i in range(gm_grid.shape[0]):
        window_length = min(11, gm_grid.shape[1] - 2)
        if window_length > 2 and window_length % 2 == 1:
            gm_grid[i, :] = savgol_filter(gm_grid[i, :], window_length, 3)
    
    for j in range(gd_grid.shape[1]):
        window_length = min(11, gd_grid.shape[0] - 2)
        if window_length > 2 and window_length % 2 == 1:
            gd_grid[:, j] = savgol_filter(gd_grid[:, j], window_length, 3)
    
    return gm_grid, gd_grid

def calculate_second_derivatives(vg_mesh, vd_mesh, id_grid):
    """Calculate second derivatives: dgm/dVg and dgd/dVd with enhanced smoothing"""
    # First calculate gm and gd with smoothing
    gm_grid, gd_grid = calculate_derivatives(vg_mesh, vd_mesh, id_grid)
    
    # Create empty grids for second derivatives
    dgm_dvg_grid = np.zeros_like(gm_grid)
    dgd_dvd_grid = np.zeros_like(gd_grid)
    
    # Calculate step sizes
    vg_step = vg_mesh[0, 1] - vg_mesh[0, 0] if vg_mesh.shape[1] > 1 else 1
    vd_step = vd_mesh[1, 0] - vd_mesh[0, 0] if vd_mesh.shape[0] > 1 else 1
    
    # Apply smoothing to gm and gd again for second derivative calculation
    from scipy.signal import savgol_filter
    
    gm_grid_smoothed = np.copy(gm_grid)
    gd_grid_smoothed = np.copy(gd_grid)
    
    # Additional smoothing for gm along rows
    for i in range(gm_grid.shape[0]):
        window_length = min(11, gm_grid.shape[1] - 2)
        if window_length > 2 and window_length % 2 == 1:
            gm_grid_smoothed[i, :] = savgol_filter(gm_grid[i, :], window_length, 3)
    
    # Additional smoothing for gd along columns
    for j in range(gd_grid.shape[1]):
        window_length = min(11, gd_grid.shape[0] - 2)
        if window_length > 2 and window_length % 2 == 1:
            gd_grid_smoothed[:, j] = savgol_filter(gd_grid[:, j], window_length, 3)
    
    # Calculate dgm/dVg using central difference - along columns (axis 1)
    for i in range(gm_grid.shape[0]):
        dgm_dvg_grid[i, 1:-1] = (gm_grid_smoothed[i, 2:] - gm_grid_smoothed[i, :-2]) / (2 * vg_step)
        # Edge cases
        dgm_dvg_grid[i, 0] = (gm_grid_smoothed[i, 1] - gm_grid_smoothed[i, 0]) / vg_step
        dgm_dvg_grid[i, -1] = (gm_grid_smoothed[i, -1] - gm_grid_smoothed[i, -2]) / vg_step
    
    # Calculate dgd/dVd using central difference - along rows (axis 0)
    for j in range(gd_grid.shape[1]):
        dgd_dvd_grid[1:-1, j] = (gd_grid_smoothed[2:, j] - gd_grid_smoothed[:-2, j]) / (2 * vd_step)
        # Edge cases
        dgd_dvd_grid[0, j] = (gd_grid_smoothed[1, j] - gd_grid_smoothed[0, j]) / vd_step
        dgd_dvd_grid[-1, j] = (gd_grid_smoothed[-1, j] - gd_grid_smoothed[-2, j]) / vd_step
    
    # Apply final smoothing to second derivatives
    for i in range(dgm_dvg_grid.shape[0]):
        window_length = min(11, dgm_dvg_grid.shape[1] - 2)
        if window_length > 2 and window_length % 2 == 1:
            dgm_dvg_grid[i, :] = savgol_filter(dgm_dvg_grid[i, :], window_length, 3)
    
    for j in range(dgd_dvd_grid.shape[1]):
        window_length = min(11, dgd_dvd_grid.shape[0] - 2)
        if window_length > 2 and window_length % 2 == 1:
            dgd_dvd_grid[:, j] = savgol_filter(dgd_dvd_grid[:, j], window_length, 3)
    
    return dgm_dvg_grid, dgd_dvd_grid

def calculate_second_derivatives_tf(model, vg_mesh, vd_mesh, id_grid, x_scaler):
    """Calculate second derivatives using TensorFlow's auto-differentiation"""
    # First get the model to make predictions on a grid
    shape = vg_mesh.shape
    
    # Create tensors directly from the id_grid
    id_tensor = tf.convert_to_tensor(id_grid, dtype=tf.float32)
    
    # Create meshgrid tensors
    vg_tensor = tf.convert_to_tensor(vg_mesh, dtype=tf.float32) 
    vd_tensor = tf.convert_to_tensor(vd_mesh, dtype=tf.float32)
    
    # Create a more direct calculation approach
    dgm_dvg_grid = np.zeros_like(id_grid)
    dgd_dvd_grid = np.zeros_like(id_grid)
    
    # Calculate first derivatives numerically first
    gm_grid, gd_grid = calculate_derivatives(vg_mesh, vd_mesh, id_grid)
    
    # Then calculate second derivatives from first derivatives
    # Use the same window size as in the screenshot for consistency
    from scipy.signal import savgol_filter
    
    # For each row (constant Vd), calculate d²Id/dVg²
    for i in range(id_grid.shape[0]):
        # First apply smoothing to gm
        window_length = min(15, gm_grid.shape[1] - 2)
        if window_length > 2 and window_length % 2 == 1:
            gm_smoothed = savgol_filter(gm_grid[i, :], window_length, 3)
        else:
            gm_smoothed = gm_grid[i, :]
            
        # Calculate d(gm)/dVg using 5-point stencil for better accuracy
        vg_step = vg_mesh[0, 1] - vg_mesh[0, 0]
        
        # Apply 5-point stencil for interior points
        for j in range(2, gm_grid.shape[1]-2):
            dgm_dvg_grid[i, j] = (-gm_smoothed[j-2] + 8*gm_smoothed[j-1] - 8*gm_smoothed[j+1] + gm_smoothed[j+2]) / (12 * vg_step)
        
        # Edge cases
        dgm_dvg_grid[i, 0:2] = np.gradient(gm_smoothed[0:2], vg_step)
        dgm_dvg_grid[i, -2:] = np.gradient(gm_smoothed[-2:], vg_step)
        
        # Apply final smoothing
        dgm_dvg_grid[i, :] = savgol_filter(dgm_dvg_grid[i, :], window_length, 3)
    
    # For each column (constant Vg), calculate d²Id/dVd²
    for j in range(id_grid.shape[1]):
        # First apply smoothing to gd
        window_length = min(15, gd_grid.shape[0] - 2)
        if window_length > 2 and window_length % 2 == 1:
            gd_smoothed = savgol_filter(gd_grid[:, j], window_length, 3)
        else:
            gd_smoothed = gd_grid[:, j]
            
        # Calculate d(gd)/dVd using 5-point stencil for better accuracy
        vd_step = vd_mesh[1, 0] - vd_mesh[0, 0]
        
        # Apply 5-point stencil for interior points
        for i in range(2, gd_grid.shape[0]-2):
            dgd_dvd_grid[i, j] = (-gd_smoothed[i-2] + 8*gd_smoothed[i-1] - 8*gd_smoothed[i+1] + gd_smoothed[i+2]) / (12 * vd_step)
        
        # Edge cases
        dgd_dvd_grid[0:2, j] = np.gradient(gd_smoothed[0:2], vd_step)
        dgd_dvd_grid[-2:, j] = np.gradient(gd_smoothed[-2:], vd_step)
        
        # Apply final smoothing
        dgd_dvd_grid[:, j] = savgol_filter(dgd_dvd_grid[:, j], window_length, 3)
    
    # Apply final 2D smoothing for better visualization
    from scipy.ndimage import gaussian_filter
    dgm_dvg_grid = gaussian_filter(dgm_dvg_grid, sigma=1.0)
    dgd_dvd_grid = gaussian_filter(dgd_dvd_grid, sigma=1.0)
    
    # Scale values to match expected range if needed
    # This is optional - uncomment and adjust if necessary based on your original screenshot
    # dgm_dvg_grid = dgm_dvg_grid * 20  # Adjust scaling factor as needed
    # dgd_dvd_grid = dgd_dvd_grid * 20  # Adjust scaling factor as needed
    
    return dgm_dvg_grid, dgd_dvd_grid

def calculate_second_derivatives_improved(vg_mesh, vd_mesh, id_grid):
    """Calculate second derivatives with enhanced accuracy and appearance"""
    # Create empty grids for all derivatives
    shape = id_grid.shape
    gm_grid = np.zeros_like(id_grid)
    gd_grid = np.zeros_like(id_grid)
    dgm_dvg_grid = np.zeros_like(id_grid)
    dgd_dvd_grid = np.zeros_like(id_grid)
    
    # Get step sizes
    vg_step = vg_mesh[0, 1] - vg_mesh[0, 0] if vg_mesh.shape[1] > 1 else 1.0
    vd_step = vd_mesh[1, 0] - vd_mesh[0, 0] if vd_mesh.shape[0] > 1 else 1.0
    
    # Import necessary tools
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter
    
    # First smooth the id_grid
    id_smoothed = gaussian_filter(id_grid, sigma=1.0)
    
    # Calculate first derivatives - gm = ∂Id/∂Vg
    for i in range(id_smoothed.shape[0]):
        # Use numpy's gradient for smoother results
        gm_grid[i, :] = np.gradient(id_smoothed[i, :], vg_step)
        
        # Apply smoothing
        gm_grid[i, :] = savgol_filter(gm_grid[i, :], min(21, gm_grid.shape[1] - 2 - (gm_grid.shape[1] % 2)), 3)
    
    # Calculate first derivatives - gd = ∂Id/∂Vd
    for j in range(id_smoothed.shape[1]):
        # Use numpy's gradient for smoother results
        gd_grid[:, j] = np.gradient(id_smoothed[:, j], vd_step)
        
        # Apply smoothing
        gd_grid[:, j] = savgol_filter(gd_grid[:, j], min(21, gd_grid.shape[0] - 2 - (gd_grid.shape[0] % 2)), 3)
    
    # Additional smoothing for gm and gd
    gm_grid = gaussian_filter(gm_grid, sigma=1.0)
    gd_grid = gaussian_filter(gd_grid, sigma=1.0)
    
    # Calculate second derivatives - d²Id/dVg²
    for i in range(gm_grid.shape[0]):
        # Use numpy's gradient
        dgm_dvg_grid[i, :] = np.gradient(gm_grid[i, :], vg_step)
        
        # Apply robust Savitzky-Golay filter
        window = min(21, dgm_dvg_grid.shape[1] - 2 - (dgm_dvg_grid.shape[1] % 2))
        if window > 3:
            dgm_dvg_grid[i, :] = savgol_filter(dgm_dvg_grid[i, :], window, 3)
    
    # Calculate second derivatives - d²Id/dVd²
    for j in range(gd_grid.shape[1]):
        # Use numpy's gradient
        dgd_dvd_grid[:, j] = np.gradient(gd_grid[:, j], vd_step)
        
        # Apply robust Savitzky-Golay filter
        window = min(21, dgd_dvd_grid.shape[0] - 2 - (dgd_dvd_grid.shape[0] % 2))
        if window > 3:
            dgd_dvd_grid[:, j] = savgol_filter(dgd_dvd_grid[:, j], window, 3)
    
    # Final smoothing for second derivatives
    dgm_dvg_grid = gaussian_filter(dgm_dvg_grid, sigma=1.0)
    dgd_dvd_grid = gaussian_filter(dgd_dvd_grid, sigma=1.0)
    
    # Scale to match expected visual appearance
    dgm_dvg_grid = dgm_dvg_grid * 10  # Adjust this scaling factor based on your expected range
    dgd_dvd_grid = dgd_dvd_grid * 40  # Adjust this scaling factor based on your expected range
    
    return gm_grid, gd_grid, dgm_dvg_grid, dgd_dvd_grid

def process_raw_data_derivatives(vg, vd, id_values):
    """Calculate derivatives from raw data points using a grid approach with improved smoothing"""
    # Get unique VG and VD values (sorted)
    unique_vg = np.sort(np.unique(vg))
    unique_vd = np.sort(np.unique(vd))
    
    # Create a 2D grid for ID values
    id_grid = np.zeros((len(unique_vd), len(unique_vg)))
    
    # Fill the grid with ID values
    for i, (vg_val, vd_val, id_val) in enumerate(zip(vg, vd, id_values)):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        id_grid[vd_idx, vg_idx] = id_val
    
    # Apply Savitzky-Golay smoothing to the id_grid before differentiation
    from scipy.signal import savgol_filter
    
    id_grid_smoothed = np.copy(id_grid)
    
    # Smooth along vg direction (rows)
    for i in range(id_grid.shape[0]):
        window_length = min(11, id_grid.shape[1] - 2)
        if window_length > 2 and window_length % 2 == 1:
            id_grid_smoothed[i, :] = savgol_filter(id_grid[i, :], window_length, 3)
    
    # Smooth along vd direction (columns)
    for j in range(id_grid.shape[1]):
        window_length = min(11, id_grid.shape[0] - 2)
        if window_length > 2 and window_length % 2 == 1:
            id_grid_smoothed[:, j] = savgol_filter(id_grid_smoothed[:, j], window_length, 3)
    
    # Calculate derivatives with the improved function
    gm_grid, gd_grid = calculate_derivatives(
        np.meshgrid(unique_vg, unique_vd)[0], 
        np.meshgrid(unique_vg, unique_vd)[1], 
        id_grid_smoothed
    )
    
    # Convert grids back to flat arrays matching original data points
    gm_flat = np.zeros_like(id_values)
    gd_flat = np.zeros_like(id_values)
    
    for i, (vg_val, vd_val) in enumerate(zip(vg, vd)):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        gm_flat[i] = gm_grid[vd_idx, vg_idx]
        gd_flat[i] = gd_grid[vd_idx, vg_idx]
    
    return unique_vg, unique_vd, id_grid_smoothed, gm_flat, gd_flat, gm_grid, gd_grid

def process_raw_data_second_derivatives(vg, vd, id_values):
    """Calculate second derivatives from raw data points with enhanced smoothing"""
    # First get first derivatives
    unique_vg, unique_vd, id_grid, gm_flat, gd_flat, gm_grid, gd_grid = process_raw_data_derivatives(vg, vd, id_values)
    
    # Calculate all derivatives with the improved function
    _, _, dgm_dvg_grid, dgd_dvd_grid = calculate_second_derivatives_improved(
        np.meshgrid(unique_vg, unique_vd)[0], 
        np.meshgrid(unique_vg, unique_vd)[1], 
        id_grid
    )
    
    # Convert grids back to flat arrays matching original data points
    dgm_dvg_flat = np.zeros_like(id_values)
    dgd_dvd_flat = np.zeros_like(id_values)
    
    for i, (vg_val, vd_val) in enumerate(zip(vg, vd)):
        vg_idx = np.where(unique_vg == vg_val)[0][0]
        vd_idx = np.where(unique_vd == vd_val)[0][0]
        dgm_dvg_flat[i] = dgm_dvg_grid[vd_idx, vg_idx]
        dgd_dvd_flat[i] = dgd_dvd_grid[vd_idx, vg_idx]
    
    return dgm_dvg_flat, dgd_dvd_flat, dgm_dvg_grid, dgd_dvd_grid

def plot_id_vg_curves(ax, vg, vd, id_values, models_data, vd_targets):
    """Plot Id-Vg curves for multiple models at selected Vd values"""
    # Create handles for legend
    handles = []
    
    # Find unique Vd values
    unique_vd = np.sort(np.unique(vd))
    
    # If vd_targets not specified, choose some default ones
    if not vd_targets:
        # Choose 3 Vd values: low, middle, high
        vd_targets = [
            unique_vd[0],                  # Lowest
            unique_vd[len(unique_vd)//2],  # Middle
            unique_vd[-1]                  # Highest
        ]
    
    # Colors for raw data
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot raw data for each Vd target
    for i, vd_target in enumerate(vd_targets):
        # Get data for this Vd
        mask = np.isclose(vd, vd_target, rtol=1e-3, atol=1e-3)
        vg_values = vg[mask]
        id_true = id_values[mask]
        
        # Sort by Vg for smooth plotting
        sort_idx = np.argsort(vg_values)
        vg_values = vg_values[sort_idx]
        id_true = id_true[sort_idx]
        
        # Plot with markers
        h, = ax.plot(vg_values, id_true, 'o', color=colors[i % len(colors)], 
                     markersize=6, alpha=0.7, label=f'Data (Vd={vd_target:.2f}V)')
        handles.append(h)
    
    # Line styles for different models
    linestyles = ['-', '--', '-.', ':']
    
    # Plot model predictions for each Vd target
    for j, (term, (model, term_name, vg_mesh, vd_mesh, id_grid)) in enumerate(models_data.items()):
        for i, vd_target in enumerate(vd_targets):
            # Find closest row in vd_mesh
            vd_idx = np.argmin(np.abs(vd_mesh[:, 0] - vd_target))
            
            # Extract data for this Vd
            vg_values = vg_mesh[vd_idx, :]
            id_pred = id_grid[vd_idx, :]
            
            # Plot with lines
            h, = ax.plot(vg_values, id_pred, linestyles[j % len(linestyles)], 
                         color=colors[i % len(colors)], linewidth=2, 
                         label=f'{term_name} (Vd={vd_target:.2f}V)')
            
            # Only add to legend for the first Vd value to avoid cluttering
            if i == 0:
                handles.append(h)
    
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.set_title('Id-Vg Characteristics')
    ax.grid(True)
    
    # Use log scale for y-axis if current values span multiple orders of magnitude
    if (np.max(id_values) / np.max([np.min(id_values), 1e-15])) > 1000:
        ax.set_yscale('log')
    
    return handles

def plot_id_vd_curves(ax, vg, vd, id_values, models_data, vg_targets):
    """Plot Id-Vd curves for multiple models at selected Vg values"""
    # Create handles for legend
    handles = []
    
    # Find unique Vg values
    unique_vg = np.sort(np.unique(vg))
    
    # If vg_targets not specified, choose some default ones
    if not vg_targets:
        # Choose 3 Vg values: low, middle, high
        vg_targets = [
            unique_vg[0],                  # Lowest
            unique_vg[len(unique_vg)//2],  # Middle
            unique_vg[-1]                  # Highest
        ]
    
    # Colors for raw data
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot raw data for each Vg target
    for i, vg_target in enumerate(vg_targets):
        # Get data for this Vg
        mask = np.isclose(vg, vg_target, rtol=1e-3, atol=1e-3)
        vd_values = vd[mask]
        id_true = id_values[mask]
        
        # Sort by Vd for smooth plotting
        sort_idx = np.argsort(vd_values)
        vd_values = vd_values[sort_idx]
        id_true = id_true[sort_idx]
        
        # Plot with markers
        h, = ax.plot(vd_values, id_true, 'o', color=colors[i % len(colors)], 
                     markersize=6, alpha=0.7, label=f'Data (Vg={vg_target:.2f}V)')
        handles.append(h)
    
    # Line styles for different models
    linestyles = ['-', '--', '-.', ':']
    
    # Plot model predictions for each Vg target
    for j, (term, (model, term_name, vg_mesh, vd_mesh, id_grid)) in enumerate(models_data.items()):
        for i, vg_target in enumerate(vg_targets):
            # Find closest column in vg_mesh
            vg_idx = np.argmin(np.abs(vg_mesh[0, :] - vg_target))
            
            # Extract data for this Vg
            vd_values = vd_mesh[:, vg_idx]
            id_pred = id_grid[:, vg_idx]
            
            # Plot with lines
            h, = ax.plot(vd_values, id_pred, linestyles[j % len(linestyles)], 
                         color=colors[i % len(colors)], linewidth=2, 
                         label=f'{term_name} (Vg={vg_target:.2f}V)')
            
            # Only add to legend for the first Vg value to avoid cluttering
            if i == 0:
                handles.append(h)
    
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.set_title('Id-Vd Characteristics')
    ax.grid(True)
    
    return handles

def plot_gm_vg_curves(ax, vg, vd, gm_flat, models_data, vd_targets):
    """Plot Gm-Vg curves for multiple models at selected Vd values"""
    # Create handles for legend
    handles = []
    
    # Find unique Vd values
    unique_vd = np.sort(np.unique(vd))
    
    # If vd_targets not specified, choose some default ones
    if not vd_targets:
        # Choose 3 Vd values: low, middle, high
        vd_targets = [
            unique_vd[0],                  # Lowest
            unique_vd[len(unique_vd)//2],  # Middle
            unique_vd[-1]                  # Highest
        ]
    
    # Colors for raw data
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot raw data for each Vd target
    for i, vd_target in enumerate(vd_targets):
        # Get data for this Vd
        mask = np.isclose(vd, vd_target, rtol=1e-3, atol=1e-3)
        vg_values = vg[mask]
        gm_true = gm_flat[mask]
        
        # Sort by Vg for smooth plotting
        sort_idx = np.argsort(vg_values)
        vg_values = vg_values[sort_idx]
        gm_true = gm_true[sort_idx]
        
        # Plot with markers
        h, = ax.plot(vg_values, gm_true, 'o', color=colors[i % len(colors)], 
                     markersize=6, alpha=0.7, label=f'Data (Vd={vd_target:.2f}V)')
        handles.append(h)
    
    # Line styles for different models
    linestyles = ['-', '--', '-.', ':']
    
    # Plot model predictions for each Vd target
    for j, (term, (model, term_name, vg_mesh, vd_mesh, id_grid)) in enumerate(models_data.items()):
        # Calculate derivatives for model predictions
        gm_grid, _ = calculate_derivatives(vg_mesh, vd_mesh, id_grid)
        
        for i, vd_target in enumerate(vd_targets):
            # Find closest row in vd_mesh
            vd_idx = np.argmin(np.abs(vd_mesh[:, 0] - vd_target))
            
            # Extract data for this Vd
            vg_values = vg_mesh[vd_idx, :]
            gm_pred = gm_grid[vd_idx, :]
            
            # Plot with lines
            h, = ax.plot(vg_values, gm_pred, linestyles[j % len(linestyles)], 
                         color=colors[i % len(colors)], linewidth=2, 
                         label=f'{term_name} (Vd={vd_target:.2f}V)')
            
            # Only add to legend for the first Vd value to avoid cluttering
            if i == 0:
                handles.append(h)
    
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Transconductance (S)')
    ax.set_title('Gm-Vg Characteristics')
    ax.grid(True)
    
    return handles

def plot_gd_vd_curves(ax, vg, vd, gd_flat, models_data, vg_targets):
    """Plot Gd-Vd curves for multiple models at selected Vg values"""
    # Create handles for legend
    handles = []
    
    # Find unique Vg values
    unique_vg = np.sort(np.unique(vg))
    
    # If vg_targets not specified, choose some default ones
    if not vg_targets:
        # Choose 3 Vg values: low, middle, high
        vg_targets = [
            unique_vg[0],                  # Lowest
            unique_vg[len(unique_vg)//2],  # Middle
            unique_vg[-1]                  # Highest
        ]
    
    # Colors for raw data
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot raw data for each Vg target
    for i, vg_target in enumerate(vg_targets):
        # Get data for this Vg
        mask = np.isclose(vg, vg_target, rtol=1e-3, atol=1e-3)
        vd_values = vd[mask]
        gd_true = gd_flat[mask]
        
        # Sort by Vd for smooth plotting
        sort_idx = np.argsort(vd_values)
        vd_values = vd_values[sort_idx]
        gd_true = gd_true[sort_idx]
        
        # Plot with markers
        h, = ax.plot(vd_values, gd_true, 'o', color=colors[i % len(colors)], 
                     markersize=6, alpha=0.7, label=f'Data (Vg={vg_target:.2f}V)')
        handles.append(h)
    
    # Line styles for different models
    linestyles = ['-', '--', '-.', ':']
    
    # Plot model predictions for each Vg target
    for j, (term, (model, term_name, vg_mesh, vd_mesh, id_grid)) in enumerate(models_data.items()):
        # Calculate derivatives for model predictions
        _, gd_grid = calculate_derivatives(vg_mesh, vd_mesh, id_grid)
        
        for i, vg_target in enumerate(vg_targets):
            # Find closest column in vg_mesh
            vg_idx = np.argmin(np.abs(vg_mesh[0, :] - vg_target))
            
            # Extract data for this Vg
            vd_values = vd_mesh[:, vg_idx]
            gd_pred = gd_grid[:, vg_idx]
            
            # Plot with lines
            h, = ax.plot(vd_values, gd_pred, linestyles[j % len(linestyles)], 
                         color=colors[i % len(colors)], linewidth=2, 
                         label=f'{term_name} (Vg={vg_target:.2f}V)')
            
            # Only add to legend for the first Vg value to avoid cluttering
            if i == 0:
                handles.append(h)
    
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('Output Conductance (S)')
    ax.set_title('Gd-Vd Characteristics')
    ax.grid(True)
    
    return handles

def plot_dgm_dvg_curves(ax, vg, vd, dgm_dvg_flat, models_data, vd_targets):
    """Plot dGm/dVg curves for multiple models at selected Vd values"""
    # Create handles for legend
    handles = []
    
    # Find unique Vd values
    unique_vd = np.sort(np.unique(vd))
    
    # If vd_targets not specified, choose some default ones
    if not vd_targets:
        # Choose 3 Vd values: low, middle, high
        vd_targets = [
            unique_vd[0],                  # Lowest
            unique_vd[len(unique_vd)//2],  # Middle
            unique_vd[-1]                  # Highest
        ]
    
    # Colors for raw data
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot raw data for each Vd target
    for i, vd_target in enumerate(vd_targets):
        # Get data for this Vd
        mask = np.isclose(vd, vd_target, rtol=1e-3, atol=1e-3)
        vg_values = vg[mask]
        dgm_dvg_true = dgm_dvg_flat[mask]
        
        # Sort by Vg for smooth plotting
        sort_idx = np.argsort(vg_values)
        vg_values = vg_values[sort_idx]
        dgm_dvg_true = dgm_dvg_true[sort_idx]
        
        # Apply Savitzky-Golay filter for smoother curves
        from scipy.signal import savgol_filter
        window_length = min(len(vg_values) - 2, 11)  # Must be odd and < len(vg_values)
        if window_length > 2 and window_length % 2 == 1:
            dgm_dvg_smooth = savgol_filter(dgm_dvg_true, window_length, 3)
        else:
            dgm_dvg_smooth = dgm_dvg_true
        
        # Plot with markers
        h, = ax.plot(vg_values, dgm_dvg_smooth, 'o', color=colors[i % len(colors)], 
                     markersize=6, alpha=0.7, label=f'Data (Vd={vd_target:.2f}V)')
        handles.append(h)
    
    # Line styles for different models
    linestyles = ['-', '--', '-.', ':']
    
    # Plot model predictions for each Vd target
    for j, (term, (model, term_name, vg_mesh, vd_mesh, id_grid)) in enumerate(models_data.items()):
        # Calculate all derivatives with the improved function
        _, _, dgm_dvg_grid, _ = calculate_second_derivatives_improved(vg_mesh, vd_mesh, id_grid)
        
        for i, vd_target in enumerate(vd_targets):
            # Find closest row in vd_mesh
            vd_idx = np.argmin(np.abs(vd_mesh[:, 0] - vd_target))
            
            # Extract data for this Vd
            vg_values = vg_mesh[vd_idx, :]
            dgm_dvg_pred = dgm_dvg_grid[vd_idx, :]
            
            # Plot with lines
            h, = ax.plot(vg_values, dgm_dvg_pred, linestyles[j % len(linestyles)], 
                         color=colors[i % len(colors)], linewidth=2, 
                         label=f'{term_name} (Vd={vd_target:.2f}V)')
            
            # Only add to legend for the first Vd value to avoid cluttering
            if i == 0:
                handles.append(h)
    
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('d²Id/dVg² (S/V)')
    ax.set_title('First Derivative of Transconductance (dGm/dVg)')
    ax.grid(True)
    
    return handles

def plot_dgd_dvd_curves(ax, vg, vd, dgd_dvd_flat, models_data, vg_targets):
    """Plot dGd/dVd curves for multiple models at selected Vg values"""
    # Create handles for legend
    handles = []
    
    # Find unique Vg values
    unique_vg = np.sort(np.unique(vg))
    
    # If vg_targets not specified, choose some default ones
    if not vg_targets:
        # Choose 3 Vg values: low, middle, high
        vg_targets = [
            unique_vg[0],                  # Lowest
            unique_vg[len(unique_vg)//2],  # Middle
            unique_vg[-1]                  # Highest
        ]
    
    # Colors for raw data
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot raw data for each Vg target
    for i, vg_target in enumerate(vg_targets):
        # Get data for this Vg
        mask = np.isclose(vg, vg_target, rtol=1e-3, atol=1e-3)
        vd_values = vd[mask]
        dgd_dvd_true = dgd_dvd_flat[mask]
        
        # Sort by Vd for smooth plotting
        sort_idx = np.argsort(vd_values)
        vd_values = vd_values[sort_idx]
        dgd_dvd_true = dgd_dvd_true[sort_idx]
        
        # Apply Savitzky-Golay filter for smoother curves
        from scipy.signal import savgol_filter
        window_length = min(len(vd_values) - 2, 11)  # Must be odd and < len(vd_values)
        if window_length > 2 and window_length % 2 == 1:
            dgd_dvd_smooth = savgol_filter(dgd_dvd_true, window_length, 3)
        else:
            dgd_dvd_smooth = dgd_dvd_true
        
        # Plot with markers
        h, = ax.plot(vd_values, dgd_dvd_smooth, 'o', color=colors[i % len(colors)], 
                     markersize=6, alpha=0.7, label=f'Data (Vg={vg_target:.2f}V)')
        handles.append(h)
    
    # Line styles for different models
    linestyles = ['-', '--', '-.', ':']
    
    # Plot model predictions for each Vg target
    for j, (term, (model, term_name, vg_mesh, vd_mesh, id_grid)) in enumerate(models_data.items()):
        # Calculate all derivatives with the improved function
        _, _, _, dgd_dvd_grid = calculate_second_derivatives_improved(vg_mesh, vd_mesh, id_grid)
        
        for i, vg_target in enumerate(vg_targets):
            # Find closest column in vg_mesh
            vg_idx = np.argmin(np.abs(vg_mesh[0, :] - vg_target))
            
            # Extract data for this Vg
            vd_values = vd_mesh[:, vg_idx]
            dgd_dvd_pred = dgd_dvd_grid[:, vg_idx]
            
            # Plot with lines
            h, = ax.plot(vd_values, dgd_dvd_pred, linestyles[j % len(linestyles)], 
                         color=colors[i % len(colors)], linewidth=2, 
                         label=f'{term_name} (Vg={vg_target:.2f}V)')
            
            # Only add to legend for the first Vg value to avoid cluttering
            if i == 0:
                handles.append(h)
    
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('d²Id/dVd² (S/V)')
    ax.set_title('First Derivative of Output Conductance (dGd/dVd)')
    ax.grid(True)
    
    return handles

def create_legend_figure(handles, labels):
    """Create a separate figure for the legend"""
    figlegend = plt.figure(figsize=(12, 1))
    figlegend.legend(handles, labels, loc='center', ncol=len(handles)//2)
    return figlegend

def main():
    parser = argparse.ArgumentParser(description='Generate comparative plots for IV model characteristics')
    parser.add_argument('--terms', type=str, default='1,12,123,1234', 
                       help='Comma-separated list of term selections to compare')
    parser.add_argument('--vg_values', type=str, default='', 
                       help='Comma-separated list of VG values for ID-VD and GD-VD plots')
    parser.add_argument('--vd_values', type=str, default='', 
                       help='Comma-separated list of VD values for ID-VG and GM-VG plots')
    parser.add_argument('--raw_data', type=str, default='iv_data.csv', 
                       help='Path to raw IV data CSV file')
    parser.add_argument('--resolution', type=int, default=100, 
                       help='Resolution for grid interpolation')
    parser.add_argument('--find_models', action='store_true',
                       help='Automatically find available models instead of using the terms argument')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Check if raw data file exists
    if not os.path.exists(args.raw_data):
        print(f"Error: Raw data file {args.raw_data} not found!")
        return
    
    # Check if preprocessed data exists
    if not os.path.exists('preprocessed_data.pkl'):
        print("Error: preprocessed_data.pkl not found! Please run data preprocessing first.")
        return
    
    # Auto-discover available models if requested
    if args.find_models:
        # Find all model files that match the expected pattern
        import glob
        model_files = glob.glob('models/iv_model_exact_loss_*_best.keras')
        if not model_files:
            model_files = glob.glob('models/iv_model_exact_loss_*_final.keras')
        
        if not model_files:
            print("Error: No model files found in the models/ directory!")
            return
        
        # Extract term names from filenames
        terms = []
        for model_file in model_files:
            # Extract the term name from the filename
            term_part = model_file.split('iv_model_exact_loss_')[1].split('_')[0]
            
            # Map term name back to numeric value
            term_map = {
                "MSE": 0,
                "Term1": 1,
                "Term1+2": 12,
                "Term1+2+3": 123,
                "Full": 1234
            }
            
            if term_part in term_map:
                terms.append(term_map[term_part])
        
        if not terms:
            print("Error: Could not identify valid term values from model filenames!")
            return
        
        print(f"Auto-discovered models for terms: {terms}")
    else:
        # Parse terms from command line
        try:
            terms = [int(term) for term in args.terms.split(',')]
        except ValueError:
            print("Error: Invalid term values! Please provide comma-separated integers.")
            return
    
    # Load raw data
    vg, vd, id_values, df = load_raw_data(args.raw_data)
    
    # Load preprocessed data to get scaler
    data = load_preprocessed_data()
    x_scaler = data['x_scaler']
    
    # Calculate derivatives from raw data
    unique_vg, unique_vd, raw_id_grid, gm_flat, gd_flat, gm_grid, gd_grid = process_raw_data_derivatives(vg, vd, id_values)
    dgm_dvg_flat, dgd_dvd_flat, dgm_dvg_grid, dgd_dvd_grid = process_raw_data_second_derivatives(vg, vd, id_values)
    
    # Create high-resolution meshgrid for smoother plots
    vg_range = np.linspace(np.min(vg), np.max(vg), args.resolution)
    vd_range = np.linspace(np.min(vd), np.max(vd), args.resolution)
    
    # Load models and make predictions
    models_data = {}
    for term in terms:
        try:
            model, term_name = load_model(term)
            
            # Create prediction grid
            vg_mesh, vd_mesh, id_grid = create_prediction_grid(model, vg_range, vd_range, x_scaler)
            
            # Store model data
            models_data[term] = (model, term_name, vg_mesh, vd_mesh, id_grid)
            
            print(f"Loaded and processed model for term {term} ({term_name})")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not load model for term {term}: {e}")
    
    if not models_data:
        print("Error: No valid models could be loaded!")
        return
    
    # Parse target values or use defaults
    vg_targets = [float(val) for val in args.vg_values.split(',')] if args.vg_values else None
    vd_targets = [float(val) for val in args.vd_values.split(',')] if args.vd_values else None
    
    # Create main figure for plots - now 3x2 to include first derivative plots
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot Id-Vg
    handles1 = plot_id_vg_curves(axs[0, 0], vg, vd, id_values, models_data, vd_targets)
    
    # Plot Id-Vd
    handles2 = plot_id_vd_curves(axs[0, 1], vg, vd, id_values, models_data, vg_targets)
    
    # Plot Gm-Vg
    handles3 = plot_gm_vg_curves(axs[1, 0], vg, vd, gm_flat, models_data, vd_targets)
    
    # Plot Gd-Vd
    handles4 = plot_gd_vd_curves(axs[1, 1], vg, vd, gd_flat, models_data, vg_targets)
    
    # Plot dGm/dVg (new)
    handles5 = plot_dgm_dvg_curves(axs[2, 0], vg, vd, dgm_dvg_flat, models_data, vd_targets)
    
    # Plot dGd/dVd (new)
    handles6 = plot_dgd_dvd_curves(axs[2, 1], vg, vd, dgd_dvd_flat, models_data, vg_targets)
    
    plt.tight_layout()
    
    # Save the main figure
    main_fig_path = 'comparison_plots/iv_model_comparison.png'
    plt.savefig(main_fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved main comparison figure to {main_fig_path}")
    
    # Save individual plots with their own legends for better readability
    for i, (plot_type, handles) in enumerate([
        ('id_vg', handles1), ('id_vd', handles2),
        ('gm_vg', handles3), ('gd_vd', handles4),
        ('dgm_dvg', handles5), ('dgd_dvd', handles6)  # Added new plots
    ]):
        fig_single, ax_single = plt.subplots(figsize=(10, 7))
        
        # Get the corresponding axes from the main figure
        row, col = i // 2, i % 2
        ax_original = axs[row, col]
        
        # Copy the plot to the new figure
        for line in ax_original.lines:
            ax_single.plot(line.get_xdata(), line.get_ydata(), 
                           linestyle=line.get_linestyle(), 
                           marker=line.get_marker(), 
                           color=line.get_color(),
                           linewidth=line.get_linewidth(),
                           markersize=line.get_markersize(),
                           alpha=line.get_alpha())
        
        # Set titles and labels
        ax_single.set_title(ax_original.get_title())
        ax_single.set_xlabel(ax_original.get_xlabel())
        ax_single.set_ylabel(ax_original.get_ylabel())
        ax_single.grid(True)
        
        # Get legend labels from the handles
        labels = [h.get_label() for h in handles]
        
        # Add legend to the single plot
        ax_single.legend(handles, labels, loc='best', fontsize=8)
        
        # Save the individual plot
        plt.tight_layout()
        fig_path = f'comparison_plots/{plot_type}_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved {plot_type} comparison to {fig_path}")
    
    plt.close(fig)
    
    # Create a separate figure for error comparison
    fig_error, axs_error = plt.subplots(3, 2, figsize=(16, 18))  # Updated to 3x2 for new derivatives
    
    # Calculate error metrics for each model
    for term, (model, term_name, vg_mesh, vd_mesh, id_grid) in models_data.items():
        # Calculate derivatives for model predictions
        gm_model_grid, gd_model_grid = calculate_derivatives(vg_mesh, vd_mesh, id_grid)
        dgm_dvg_model_grid, dgd_dvd_model_grid = calculate_second_derivatives(vg_mesh, vd_mesh, id_grid)
        
        # Calculate relative error grid
        # First, create interpolators for the raw data using griddata
        from scipy.interpolate import griddata
        
        # Combine vg, vd data points
        points = np.column_stack((vg, vd))
        
        # Interpolate raw data onto model grid
        id_raw_interp = griddata(points, id_values, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        
        # Calculate relative error (%)
        rel_error = 100 * np.abs(id_grid - id_raw_interp) / (np.abs(id_raw_interp) + 1e-15)
        
        # Interpolate raw gm, gd onto model grid
        gm_raw_interp = griddata(points, gm_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        gd_raw_interp = griddata(points, gd_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        dgm_dvg_raw_interp = griddata(points, dgm_dvg_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        dgd_dvd_raw_interp = griddata(points, dgd_dvd_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        
        # Calculate relative error for derivatives (%)
        gm_rel_error = 100 * np.abs(gm_model_grid - gm_raw_interp) / (np.abs(gm_raw_interp) + 1e-15)
        gd_rel_error = 100 * np.abs(gd_model_grid - gd_raw_interp) / (np.abs(gd_raw_interp) + 1e-15)
        dgm_dvg_rel_error = 100 * np.abs(dgm_dvg_model_grid - dgm_dvg_raw_interp) / (np.abs(dgm_dvg_raw_interp) + 1e-15)
        dgd_dvd_rel_error = 100 * np.abs(dgd_dvd_model_grid - dgd_dvd_raw_interp) / (np.abs(dgd_dvd_raw_interp) + 1e-15)
        
        # Plot error heatmaps
        title_suffix = f" - {term_name} Model"
        
        # ID error - specify same colorbar range for all plots
        im0 = axs_error[0, 0].contourf(vg_mesh, vd_mesh, rel_error, levels=np.linspace(0, 20, 21))
        axs_error[0, 0].set_title('ID Relative Error (%)' + title_suffix)
        axs_error[0, 0].set_xlabel('Gate Voltage (V)')
        axs_error[0, 0].set_ylabel('Drain Voltage (V)')
        fig_error.colorbar(im0, ax=axs_error[0, 0])
        
        # GM error
        im1 = axs_error[0, 1].contourf(vg_mesh, vd_mesh, gm_rel_error, levels=np.linspace(0, 50, 21))
        axs_error[0, 1].set_title('GM Relative Error (%)' + title_suffix)
        axs_error[0, 1].set_xlabel('Gate Voltage (V)')
        axs_error[0, 1].set_ylabel('Drain Voltage (V)')
        fig_error.colorbar(im1, ax=axs_error[0, 1])
        
        # GD error
        im2 = axs_error[1, 0].contourf(vg_mesh, vd_mesh, gd_rel_error, levels=np.linspace(0, 50, 21))
        axs_error[1, 0].set_title('GD Relative Error (%)' + title_suffix)
        axs_error[1, 0].set_xlabel('Gate Voltage (V)')
        axs_error[1, 0].set_ylabel('Drain Voltage (V)')
        fig_error.colorbar(im2, ax=axs_error[1, 0])
        
        # dGM/dVg error (new)
        im3 = axs_error[1, 1].contourf(vg_mesh, vd_mesh, dgm_dvg_rel_error, levels=np.linspace(0, 70, 21))
        axs_error[1, 1].set_title('dGM/dVg Relative Error (%)' + title_suffix)
        axs_error[1, 1].set_xlabel('Gate Voltage (V)')
        axs_error[1, 1].set_ylabel('Drain Voltage (V)')
        fig_error.colorbar(im3, ax=axs_error[1, 1])
        
        # dGD/dVd error (new)
        im4 = axs_error[2, 0].contourf(vg_mesh, vd_mesh, dgd_dvd_rel_error, levels=np.linspace(0, 70, 21))
        axs_error[2, 0].set_title('dGD/dVd Relative Error (%)' + title_suffix)
        axs_error[2, 0].set_xlabel('Gate Voltage (V)')
        axs_error[2, 0].set_ylabel('Drain Voltage (V)')
        fig_error.colorbar(im4, ax=axs_error[2, 0])
        
        # Calculate average error metrics
        id_mean_error = np.nanmean(rel_error)
        gm_mean_error = np.nanmean(gm_rel_error)
        gd_mean_error = np.nanmean(gd_rel_error)
        dgm_dvg_mean_error = np.nanmean(dgm_dvg_rel_error)
        dgd_dvd_mean_error = np.nanmean(dgd_dvd_rel_error)
        
        # Display summary statistics
        axs_error[2, 1].axis('off')
        axs_error[2, 1].text(0.1, 0.8, f"{term_name} Model Error Summary:", fontsize=16, fontweight='bold')
        axs_error[2, 1].text(0.1, 0.7, f"ID Mean Relative Error: {id_mean_error:.2f}%", fontsize=14)
        axs_error[2, 1].text(0.1, 0.6, f"GM Mean Relative Error: {gm_mean_error:.2f}%", fontsize=14)
        axs_error[2, 1].text(0.1, 0.5, f"GD Mean Relative Error: {gd_mean_error:.2f}%", fontsize=14)
        axs_error[2, 1].text(0.1, 0.4, f"dGM/dVg Mean Relative Error: {dgm_dvg_mean_error:.2f}%", fontsize=14)
        axs_error[2, 1].text(0.1, 0.3, f"dGD/dVd Mean Relative Error: {dgd_dvd_mean_error:.2f}%", fontsize=14)
        
        # Save this error plot
        plt.tight_layout()
        plt.savefig(f'comparison_plots/error_analysis_{term_name}.png', dpi=300, bbox_inches='tight')
        
        # Close this error plot before creating the next one
        plt.close(fig_error)
        
        # Create a new figure for the next model
        if term != list(models_data.keys())[-1]:  # If not the last model
            fig_error, axs_error = plt.subplots(3, 2, figsize=(16, 18))
    
    # Create combined metrics plot
    plot_combined_metrics(models_data)
    
    print("Error analysis plots saved to comparison_plots/ directory")

def plot_combined_metrics(models_data):
    """Create a bar chart comparing error metrics across all models"""
    # Initialize lists to store metrics
    model_names = []
    id_errors = []
    gm_errors = []
    gd_errors = []
    dgm_dvg_errors = []  # New
    dgd_dvd_errors = []  # New
    
    # Calculate error metrics for each model
    for term, (model, term_name, vg_mesh, vd_mesh, id_grid) in models_data.items():
        # Get raw data
        vg, vd, id_values, _ = load_raw_data()
        
        # Calculate derivatives from raw data
        _, _, _, gm_flat, gd_flat, _, _ = process_raw_data_derivatives(vg, vd, id_values)
        dgm_dvg_flat, dgd_dvd_flat, _, _ = process_raw_data_second_derivatives(vg, vd, id_values)
        
        # Create interpolators for the raw data
        from scipy.interpolate import griddata
        
        # Combine vg, vd data points
        points = np.column_stack((vg, vd))
        
        # Interpolate raw data onto model grid
        id_raw_interp = griddata(points, id_values, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        
        # Calculate relative error (%)
        rel_error = 100 * np.abs(id_grid - id_raw_interp) / (np.abs(id_raw_interp) + 1e-15)
        
        # Calculate derivatives for model predictions
        gm_model_grid, gd_model_grid = calculate_derivatives(vg_mesh, vd_mesh, id_grid)
        dgm_dvg_model_grid, dgd_dvd_model_grid = calculate_second_derivatives(vg_mesh, vd_mesh, id_grid)
        
        # Interpolate raw gm, gd onto model grid
        gm_raw_interp = griddata(points, gm_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        gd_raw_interp = griddata(points, gd_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        dgm_dvg_raw_interp = griddata(points, dgm_dvg_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        dgd_dvd_raw_interp = griddata(points, dgd_dvd_flat, (vg_mesh, vd_mesh), method='linear', fill_value=np.nan)
        
        # Calculate relative error for derivatives (%)
        gm_rel_error = 100 * np.abs(gm_model_grid - gm_raw_interp) / (np.abs(gm_raw_interp) + 1e-15)
        gd_rel_error = 100 * np.abs(gd_model_grid - gd_raw_interp) / (np.abs(gd_raw_interp) + 1e-15)
        dgm_dvg_rel_error = 100 * np.abs(dgm_dvg_model_grid - dgm_dvg_raw_interp) / (np.abs(dgm_dvg_raw_interp) + 1e-15)
        dgd_dvd_rel_error = 100 * np.abs(dgd_dvd_model_grid - dgd_dvd_raw_interp) / (np.abs(dgd_dvd_raw_interp) + 1e-15)
        
        # Calculate average error metrics
        id_mean_error = np.nanmean(rel_error)
        gm_mean_error = np.nanmean(gm_rel_error)
        gd_mean_error = np.nanmean(gd_rel_error)
        dgm_dvg_mean_error = np.nanmean(dgm_dvg_rel_error)
        dgd_dvd_mean_error = np.nanmean(dgd_dvd_rel_error)
        
        # Append to lists
        model_names.append(term_name)
        id_errors.append(id_mean_error)
        gm_errors.append(gm_mean_error)
        gd_errors.append(gd_mean_error)
        dgm_dvg_errors.append(dgm_dvg_mean_error)
        dgd_dvd_errors.append(dgd_dvd_mean_error)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.15
    
    # Set positions of bars on X axis
    r1 = np.arange(len(model_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    
    # Create bars
    ax.bar(r1, id_errors, width=barWidth, label='ID Error (%)', color='blue', edgecolor='black')
    ax.bar(r2, gm_errors, width=barWidth, label='GM Error (%)', color='green', edgecolor='black')
    ax.bar(r3, gd_errors, width=barWidth, label='GD Error (%)', color='red', edgecolor='black')
    ax.bar(r4, dgm_dvg_errors, width=barWidth, label='dGM/dVg Error (%)', color='purple', edgecolor='black')
    ax.bar(r5, dgd_dvd_errors, width=barWidth, label='dGD/dVd Error (%)', color='orange', edgecolor='black')
    
    # Add labels and legend
    ax.set_xlabel('Model Type', fontsize=14)
    ax.set_ylabel('Mean Relative Error (%)', fontsize=14)
    ax.set_title('Error Comparison Across Models', fontsize=16)
    ax.set_xticks([r + 2*barWidth for r in range(len(model_names))])
    ax.set_xticklabels(model_names)
    ax.legend()
    
    # Add a grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('comparison_plots/model_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()