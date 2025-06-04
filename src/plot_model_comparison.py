import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

def load_models(model_paths, model_names):
    """Load specific models based on provided paths"""
    models = {}
    
    for i, path in enumerate(model_paths):
        if os.path.exists(path):
            try:
                # Load the model without recompiling
                model = tf.keras.models.load_model(path, compile=False)
                name = model_names[i]
                models[name] = model
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Error loading model {path}: {str(e)}")
        else:
            print(f"Model file not found: {path}")
    
    return models

def load_original_data(file_path='iv_data.csv'):
    """Load the original I-V data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original data file {file_path} not found")
    
    df = pd.read_csv(file_path)
    
    # Verify the required columns exist
    required_cols = ['Vg', 'Vd', 'Id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {file_path}")
    
    return df

def load_preprocessed_data(file_path='preprocessed_data.pkl'):
    """Load the preprocessed data if available"""
    if not os.path.exists(file_path):
        print(f"Warning: Preprocessed data file {file_path} not found")
        return None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

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

def calculate_gm(model, vg, vd, delta=0.001, x_scaler=None):
    """Calculate transconductance (gm = dId/dVg) using numerical differentiation"""
    id1 = predict_current(model, vg, vd, x_scaler)
    id2 = predict_current(model, vg + delta, vd, x_scaler)
    gm = (id2 - id1) / delta
    return gm

def calculate_gd(model, vg, vd, delta=0.001, x_scaler=None):
    """Calculate output conductance (gd = dId/dVd) using numerical differentiation"""
    id1 = predict_current(model, vg, vd, x_scaler)
    id2 = predict_current(model, vg, vd + delta, x_scaler)
    gd = (id2 - id1) / delta
    return gd

def calculate_error_metrics(original_data, model, x_scaler=None):
    """Calculate error metrics for the model"""
    vg = original_data['Vg'].values
    vd = original_data['Vd'].values
    id_true = original_data['Id'].values
    
    # Predict drain currents
    id_pred = predict_current(model, vg, vd, x_scaler)
    
    # Calculate MAPE
    mape = np.mean(np.abs((id_true - id_pred) / id_true)) * 100
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((id_true - id_pred) ** 2))
    
    return {
        'MAPE': mape,
        'RMSE': rmse
    }

def create_comparison_plots(models, original_data, output_dir='exact_loss_comparison', preprocessed_data=None):
    """Create comprehensive comparison plots for exact loss models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the input scaler from preprocessed data if available
    x_scaler = None
    if preprocessed_data and 'x_scaler' in preprocessed_data:
        x_scaler = preprocessed_data['x_scaler']
    
    # Get unique Vg and Vd values from original data
    unique_vg = np.sort(np.unique(original_data['Vg'].values))
    unique_vd = np.sort(np.unique(original_data['Vd'].values))
    
    # Create color map for models
    colors = plt.cm.tab10.colors
    model_colors = {name: colors[i % len(colors)] for i, name in enumerate(models.keys())}
    
    # Calculate error metrics for all models
    metrics = {}
    for name, model in models.items():
        metrics[name] = calculate_error_metrics(original_data, model, x_scaler)
    
    # Create figure with 2x3 grid for plots
    fig = plt.figure(figsize=(18, 12))
    grid = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. Id-Vd curves for multiple Vg values (top-left)
    ax1 = fig.add_subplot(grid[0, 0])
    plot_id_vd_comparison(ax1, models, original_data, unique_vg, unique_vd, model_colors, x_scaler)
    
    # 2. Id-Vg curves for multiple Vd values (top-middle)
    ax2 = fig.add_subplot(grid[0, 1])
    plot_id_vg_comparison(ax2, models, original_data, unique_vg, unique_vd, model_colors, x_scaler)
    
    # 3. gm-Vg curves for a specific Vd (top-right)
    ax3 = fig.add_subplot(grid[0, 2])
    plot_gm_vg_comparison(ax3, models, original_data, unique_vg, unique_vd, model_colors, x_scaler)
    
    # 4. gd-Vd curves for a specific Vg (bottom-left)
    ax4 = fig.add_subplot(grid[1, 0])
    plot_gd_vd_comparison(ax4, models, original_data, unique_vg, unique_vd, model_colors, x_scaler)
    
    # 5. Error metrics comparison (bottom-middle)
    ax5 = fig.add_subplot(grid[1, 1])
    plot_error_metrics(ax5, metrics)
    
    # 6. Semi-log Id-Vg plot (bottom-right)
    ax6 = fig.add_subplot(grid[1, 2])
    plot_semilogy_id_vg(ax6, models, original_data, unique_vg, unique_vd, model_colors, x_scaler)
    
    # Add title and adjust layout
    plt.suptitle('Neural Network Model Comparison with Different Loss Functions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'exact_loss_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional plots for specific operating points
    create_operating_point_plots(models, original_data, unique_vg, unique_vd, model_colors, output_dir, x_scaler)
    
    # Save error metrics to CSV
    save_metrics_to_csv(metrics, os.path.join(output_dir, 'exact_loss_model_metrics.csv'))
    
    print(f"All comparison plots saved to {output_dir}")

def plot_id_vd_comparison(ax, models, original_data, unique_vg, unique_vd, model_colors, x_scaler=None):
    """Plot Id-Vd characteristics for selected Vg values"""
    # Choose a few representative Vg values
    if len(unique_vg) >= 4:
        vg_to_plot = [0.2, 0.4, 0.6, 0.8]  # Specified values
    else:
        vg_to_plot = unique_vg
    
    # Generate smooth Vd values for model predictions
    vd_smooth = np.linspace(np.min(unique_vd), np.max(unique_vd), 100)
    
    # Create different line styles for different Vg values
    line_styles = ['-', '--', '-.', ':']
    
    # Plot original data and model predictions for each Vg value
    for i, vg_target in enumerate(vg_to_plot):
        style = line_styles[i % len(line_styles)]
        
        # Extract original data for this Vg
        mask = np.isclose(original_data['Vg'].values, vg_target, atol=1e-2)
        orig_vd = original_data['Vd'].values[mask]
        orig_id = original_data['Id'].values[mask]
        
        # Sort by Vd for proper connection
        if len(orig_vd) > 0:
            sort_idx = np.argsort(orig_vd)
            orig_vd = orig_vd[sort_idx]
            orig_id = orig_id[sort_idx]
            
            # Plot original data points
            step = max(1, len(orig_vd) // 15)  # Subsample for cleaner plot
            ax.plot(orig_vd[::step], orig_id[::step], 'o', color='black', markersize=4, alpha=0.7,
                    label=f'Data (Vg={vg_target:.2f}V)' if i == 0 else f'Data (Vg={vg_target:.2f}V)')
        
        # Plot model predictions for this Vg
        vg_smooth = np.ones_like(vd_smooth) * vg_target
        
        for j, (model_name, model) in enumerate(models.items()):
            id_pred = predict_current(model, vg_smooth, vd_smooth, x_scaler)
            
            # Only add to legend for the first Vg
            if i == 0:
                ax.plot(vd_smooth, id_pred, style, color=model_colors[model_name], linewidth=2,
                        label=f'{model_name} model')
            else:
                ax.plot(vd_smooth, id_pred, style, color=model_colors[model_name], linewidth=2)
    
    ax.set_title('Id-Vd Characteristics')
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Add legend with smaller font size and only showing each model once
    handles, labels = ax.get_legend_handles_labels()
    # Filter to show one entry per model
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, fontsize=8, loc='best')

def plot_id_vg_comparison(ax, models, original_data, unique_vg, unique_vd, model_colors, x_scaler=None):
    """Plot Id-Vg characteristics for selected Vd values"""
    # Choose a few representative Vd values
    if len(unique_vd) >= 3:
        vd_to_plot = [0.1, 0.4, 0.8]  # Specified values
    else:
        vd_to_plot = unique_vd
    
    # Generate smooth Vg values for model predictions
    vg_smooth = np.linspace(np.min(unique_vg), np.max(unique_vg), 100)
    
    # Create different line styles for different Vd values
    line_styles = ['-', '--', '-.']
    
    # Plot original data and model predictions for each Vd value
    for i, vd_target in enumerate(vd_to_plot):
        style = line_styles[i % len(line_styles)]
        
        # Extract original data for this Vd
        mask = np.isclose(original_data['Vd'].values, vd_target, atol=1e-2)
        orig_vg = original_data['Vg'].values[mask]
        orig_id = original_data['Id'].values[mask]
        
        # Sort by Vg for proper connection
        if len(orig_vg) > 0:
            sort_idx = np.argsort(orig_vg)
            orig_vg = orig_vg[sort_idx]
            orig_id = orig_id[sort_idx]
            
            # Plot original data points
            step = max(1, len(orig_vg) // 15)  # Subsample for cleaner plot
            ax.plot(orig_vg[::step], orig_id[::step], 'o', color='black', markersize=4, alpha=0.7,
                    label=f'Data (Vd={vd_target:.2f}V)' if i == 0 else f'Data (Vd={vd_target:.2f}V)')
        
        # Plot model predictions for this Vd
        vd_smooth = np.ones_like(vg_smooth) * vd_target
        
        for j, (model_name, model) in enumerate(models.items()):
            id_pred = predict_current(model, vg_smooth, vd_smooth, x_scaler)
            
            # Only add to legend for the first Vd
            if i == 0:
                ax.plot(vg_smooth, id_pred, style, color=model_colors[model_name], linewidth=2,
                        label=f'{model_name} model')
            else:
                ax.plot(vg_smooth, id_pred, style, color=model_colors[model_name], linewidth=2)
    
    ax.set_title('Id-Vg Characteristics')
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Add legend with smaller font size
    handles, labels = ax.get_legend_handles_labels()
    # Filter to show one entry per model
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, fontsize=8, loc='best')

def plot_gm_vg_comparison(ax, models, original_data, unique_vg, unique_vd, model_colors, x_scaler=None):
    """Plot gm-Vg characteristics for a specific Vd value"""
    # Choose a middle Vd value
    vd_target = unique_vd[len(unique_vd)//2]
    
    # Generate smooth Vg values for model predictions
    vg_smooth = np.linspace(np.min(unique_vg), np.max(unique_vg), 100)
    vd_smooth = np.ones_like(vg_smooth) * vd_target
    
    # Calculate and plot gm for original data (if sufficient data points)
    try:
        mask = np.isclose(original_data['Vd'].values, vd_target, atol=1e-2)
        orig_vg = original_data['Vg'].values[mask]
        orig_id = original_data['Id'].values[mask]
        
        if len(orig_vg) > 5:  # Need enough points for derivative calculation
            # Sort by Vg
            sort_idx = np.argsort(orig_vg)
            orig_vg = orig_vg[sort_idx]
            orig_id = orig_id[sort_idx]
            
            # Smooth the data to reduce noise
            window_size = min(5, len(orig_vg) - (len(orig_vg) % 2 == 0))
            if window_size >= 3:
                orig_id_smooth = savgol_filter(orig_id, window_size, 2)
                
                # Calculate numerical derivative
                gm_orig = np.gradient(orig_id_smooth, orig_vg)
                
                # Smooth the derivative
                gm_orig_smooth = savgol_filter(gm_orig, window_size, 1)
                
                # Plot original gm data
                step = max(1, len(orig_vg) // 12)  # Subsample for cleaner plot
                ax.plot(orig_vg[::step], gm_orig_smooth[::step], 'o', color='black', markersize=4, alpha=0.7,
                        label=f'Data')
    except Exception as e:
        print(f"Warning: Could not calculate gm from raw data: {str(e)}")
    
    # Plot gm for each model
    for model_name, model in models.items():
        gm_pred = calculate_gm(model, vg_smooth, vd_smooth, x_scaler=x_scaler)
        ax.plot(vg_smooth, gm_pred, '-', color=model_colors[model_name], linewidth=2,
                label=f'{model_name} model')
    
    ax.set_title(f'Transconductance (gm=Vg) at Vd={vd_target:.2f}V')
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Transconductance (S)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(fontsize=8, loc='best')

def plot_gd_vd_comparison(ax, models, original_data, unique_vg, unique_vd, model_colors, x_scaler=None):
    """Plot gd-Vd characteristics for a specific Vg value"""
    # Choose a middle Vg value
    vg_target = unique_vg[len(unique_vg)//2]
    
    # Generate smooth Vd values for model predictions
    vd_smooth = np.linspace(np.min(unique_vd), np.max(unique_vd), 100)
    vg_smooth = np.ones_like(vd_smooth) * vg_target
    
    # Calculate and plot gd for original data (if sufficient data points)
    try:
        mask = np.isclose(original_data['Vg'].values, vg_target, atol=1e-2)
        orig_vd = original_data['Vd'].values[mask]
        orig_id = original_data['Id'].values[mask]
        
        if len(orig_vd) > 5:  # Need enough points for derivative calculation
            # Sort by Vd
            sort_idx = np.argsort(orig_vd)
            orig_vd = orig_vd[sort_idx]
            orig_id = orig_id[sort_idx]
            
            # Smooth the data to reduce noise
            window_size = min(5, len(orig_vd) - (len(orig_vd) % 2 == 0))
            if window_size >= 3:
                orig_id_smooth = savgol_filter(orig_id, window_size, 2)
                
                # Calculate numerical derivative
                gd_orig = np.gradient(orig_id_smooth, orig_vd)
                
                # Smooth the derivative
                gd_orig_smooth = savgol_filter(gd_orig, window_size, 1)
                
                # Plot original gd data
                step = max(1, len(orig_vd) // 12)  # Subsample for cleaner plot
                ax.plot(orig_vd[::step], gd_orig_smooth[::step], 'o', color='black', markersize=4, alpha=0.7,
                        label=f'Data')
    except Exception as e:
        print(f"Warning: Could not calculate gd from raw data: {str(e)}")
    
    # Plot gd for each model
    for model_name, model in models.items():
        gd_pred = calculate_gd(model, vg_smooth, vd_smooth, x_scaler=x_scaler)
        ax.plot(vd_smooth, gd_pred, '-', color=model_colors[model_name], linewidth=2,
                label=f'{model_name} model')
    
    ax.set_title(f'Output Conductance (gd-Vd) at Vg={vg_target:.2f}V')
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('Output Conductance (S)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(fontsize=8, loc='best')

def plot_error_metrics(ax, metrics):
    """Plot error metrics for all models"""
    model_names = list(metrics.keys())
    mape_values = [metrics[name]['MAPE'] for name in model_names]
    rmse_values = [metrics[name]['RMSE'] for name in model_names]
    
    # Normalize RMSE for plotting
    max_rmse = max(rmse_values)
    normalized_rmse = [val / max_rmse * 100 for val in rmse_values]
    
    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, mape_values, width, label='MAPE (%)', color='skyblue')
    bars2 = ax.bar(x + width/2, normalized_rmse, width, label='Normalized RMSE', color='salmon')
    
    # Add text labels on bars
    for bar, value in zip(bars1, mape_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
    
    for bar, value, orig_value in zip(bars2, normalized_rmse, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{orig_value:.2e}', ha='center', va='bottom', fontsize=8)
    
    ax.set_title('Model Error Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Error Metric Value')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

def plot_semilogy_id_vg(ax, models, original_data, unique_vg, unique_vd, model_colors, x_scaler=None):
    """Plot Id-Vg characteristics with logarithmic y-axis"""
    # Choose a middle Vd value
    vd_target = unique_vd[len(unique_vd)//2]
    
    # Generate smooth Vg values for model predictions
    vg_smooth = np.linspace(np.min(unique_vg), np.max(unique_vg), 100)
    vd_smooth = np.ones_like(vg_smooth) * vd_target
    
    # Extract original data for this Vd
    mask = np.isclose(original_data['Vd'].values, vd_target, atol=1e-2)
    orig_vg = original_data['Vg'].values[mask]
    orig_id = original_data['Id'].values[mask]
    
    # Sort by Vg for proper connection
    if len(orig_vg) > 0:
        sort_idx = np.argsort(orig_vg)
        orig_vg = orig_vg[sort_idx]
        orig_id = orig_id[sort_idx]
        
        # Plot original data points
        step = max(1, len(orig_vg) // 15)  # Subsample for cleaner plot
        ax.semilogy(orig_vg[::step], orig_id[::step], 'o', color='black', markersize=4, alpha=0.7,
                label=f'Data')
    
    # Plot model predictions
    for model_name, model in models.items():
        id_pred = predict_current(model, vg_smooth, vd_smooth, x_scaler)
        ax.semilogy(vg_smooth, id_pred, '-', color=model_colors[model_name], linewidth=2,
                label=f'{model_name} model')
    
    ax.set_title(f'Id-Vg Characteristics (Log Scale) at Vd={vd_target:.2f}V')
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='best')

def create_operating_point_plots(models, original_data, unique_vg, unique_vd, model_colors, output_dir, x_scaler=None):
    """Create plots for specific operating points to show detailed comparison"""
    # Choose a specific operating point (middle values)
    vg_target = unique_vg[len(unique_vg)//2]
    vd_target = unique_vd[len(unique_vd)//2]
    
    # Create figure with 2x2 grid for detailed operating point comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Id-Vd at specific Vg (top-left)
    ax = axs[0, 0]
    # Extract original data
    mask = np.isclose(original_data['Vg'].values, vg_target, atol=1e-2)
    orig_vd = original_data['Vd'].values[mask]
    orig_id = original_data['Id'].values[mask]
    
    # Sort and plot original data
    if len(orig_vd) > 0:
        sort_idx = np.argsort(orig_vd)
        orig_vd = orig_vd[sort_idx]
        orig_id = orig_id[sort_idx]
        ax.plot(orig_vd, orig_id, 'o', color='black', markersize=4, label='Original Data')
    
    # Generate smooth Vd values
    vd_smooth = np.linspace(np.min(unique_vd), np.max(unique_vd), 100)
    vg_smooth = np.ones_like(vd_smooth) * vg_target
    
    # Plot model predictions
    for model_name, model in models.items():
        id_pred = predict_current(model, vg_smooth, vd_smooth, x_scaler)
        ax.plot(vd_smooth, id_pred, '-', color=model_colors[model_name], linewidth=2, label=f'{model_name}')
    
    ax.set_title(f'Id-Vd Characteristics at Vg={vg_target:.2f}V')
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(fontsize=9)
    
    # 2. Id-Vg at specific Vd (top-right)
    ax = axs[0, 1]
    # Extract original data
    mask = np.isclose(original_data['Vd'].values, vd_target, atol=1e-2)
    orig_vg = original_data['Vg'].values[mask]
    orig_id = original_data['Id'].values[mask]
    
    # Sort and plot original data
    if len(orig_vg) > 0:
        sort_idx = np.argsort(orig_vg)
        orig_vg = orig_vg[sort_idx]
        orig_id = orig_id[sort_idx]
        ax.plot(orig_vg, orig_id, 'o', color='black', markersize=4, label='Original Data')
    
    # Generate smooth Vg values
    vg_smooth = np.linspace(np.min(unique_vg), np.max(unique_vg), 100)
    vd_smooth = np.ones_like(vg_smooth) * vd_target
    
    # Plot model predictions
    for model_name, model in models.items():
        id_pred = predict_current(model, vg_smooth, vd_smooth, x_scaler)
        ax.plot(vg_smooth, id_pred, '-', color=model_colors[model_name], linewidth=2, label=f'{model_name}')
    
    ax.set_title(f'Id-Vg Characteristics at Vd={vd_target:.2f}V')
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Drain Current (A)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(fontsize=9)
    
    # 3. gm-Vg at specific Vd (bottom-left)
    ax = axs[1, 0]
    # Generate smooth Vg values
    vg_smooth = np.linspace(np.min(unique_vg), np.max(unique_vg), 100)
    vd_smooth = np.ones_like(vg_smooth) * vd_target
    
    # Plot model predictions
    for model_name, model in models.items():
        gm_pred = calculate_gm(model, vg_smooth, vd_smooth, x_scaler=x_scaler)
        ax.plot(vg_smooth, gm_pred, '-', color=model_colors[model_name], linewidth=2, label=f'{model_name}')
    
    ax.set_title(f'Transconductance (gm) at Vd={vd_target:.2f}V')
    ax.set_title(f'Transconductance (gm) at Vd={vd_target:.2f}V')
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Transconductance (S)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(fontsize=9)
    
    # 4. gd-Vd at specific Vg (bottom-right)
    ax = axs[1, 1]
    # Generate smooth Vd values
    vd_smooth = np.linspace(np.min(unique_vd), np.max(unique_vd), 100)
    vg_smooth = np.ones_like(vd_smooth) * vg_target
    
    # Plot model predictions
    for model_name, model in models.items():
        gd_pred = calculate_gd(model, vg_smooth, vd_smooth, x_scaler=x_scaler)
        ax.plot(vd_smooth, gd_pred, '-', color=model_colors[model_name], linewidth=2, label=f'{model_name}')
    
    ax.set_title(f'Output Conductance (gd) at Vg={vg_target:.2f}V')
    ax.set_xlabel('Drain Voltage (V)')
    ax.set_ylabel('Output Conductance (S)')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'exact_loss_operating_point_vg{vg_target:.2f}_vd{vd_target:.2f}.png'), dpi=300)
    plt.close()
    
    # Create specific plots for subthreshold region
    create_subthreshold_plots(models, original_data, unique_vg, unique_vd, model_colors, output_dir, x_scaler)

def create_subthreshold_plots(models, original_data, unique_vg, unique_vd, model_colors, output_dir, x_scaler=None):
    """Create plots focusing on the subthreshold region"""
    # Choose a lower Vd value for subthreshold analysis
    vd_index = max(0, len(unique_vd) // 3)  # Choose a lower Vd
    vd_target = unique_vd[vd_index]
    
    # Focus on lower Vg values for subthreshold
    vg_min = np.min(unique_vg)
    vg_max = np.mean(unique_vg)  # Use lower half of Vg range
    
    # Create a figure for subthreshold Id-Vg plot (log scale)
    plt.figure(figsize=(10, 8))
    
    # Extract original data for this Vd
    mask = np.isclose(original_data['Vd'].values, vd_target, atol=1e-2)
    orig_vg = original_data['Vg'].values[mask]
    orig_id = original_data['Id'].values[mask]
    
    # Filter for subthreshold region
    sub_mask = (orig_vg >= vg_min) & (orig_vg <= vg_max)
    orig_vg = orig_vg[sub_mask]
    orig_id = orig_id[sub_mask]
    
    # Sort and plot original data
    if len(orig_vg) > 0:
        sort_idx = np.argsort(orig_vg)
        orig_vg = orig_vg[sort_idx]
        orig_id = orig_id[sort_idx]
        plt.semilogy(orig_vg, orig_id, 'o', color='black', markersize=4, label='Original Data')
    
    # Generate smooth Vg values in subthreshold region
    vg_smooth = np.linspace(vg_min, vg_max, 100)
    vd_smooth = np.ones_like(vg_smooth) * vd_target
    
    # Plot model predictions
    for model_name, model in models.items():
        id_pred = predict_current(model, vg_smooth, vd_smooth, x_scaler)
        plt.semilogy(vg_smooth, id_pred, '-', color=model_colors[model_name], linewidth=2, label=f'{model_name}')
    
    plt.title(f'Subthreshold Id-Vg Characteristics (Log Scale) at Vd={vd_target:.2f}V')
    plt.xlabel('Gate Voltage (V)')
    plt.ylabel('Drain Current (A)')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exact_loss_subthreshold_id_vg.png'), dpi=300)
    plt.close()

def calculate_subthreshold_slope(model, vg, vd, x_scaler=None):
    """Calculate subthreshold slope (mV/decade)"""
    # Calculate two currents at slightly different Vg values
    id1 = predict_current(model, vg, vd, x_scaler)
    id2 = predict_current(model, vg + 0.001, vd, x_scaler)
    
    # Calculate slope in mV/decade
    slope = 1000 * 0.001 / np.log10(id2/id1)
    
    return slope

def save_metrics_to_csv(metrics, filepath):
    """Save model metrics to a CSV file"""
    with open(filepath, 'w') as f:
        # Write header
        f.write('Model,MAPE (%),RMSE (A)\n')
        
        # Write data
        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name},{model_metrics['MAPE']:.4f},{model_metrics['RMSE']:.6e}\n")
    
    print(f"Metrics saved to {filepath}")

def compare_term_models(term_numbers, iv_data_path='iv_data.csv', preprocessed_data_path='preprocessed_data.pkl', output_dir='exact_loss_term_comparison'):
    """Compare models trained with different loss term combinations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original data
    original_data = load_original_data(iv_data_path)
    if original_data is None:
        return
    
    # Load preprocessed data
    preprocessed_data = load_preprocessed_data(preprocessed_data_path)
    
    # Prepare model paths and names
    model_paths = []
    model_names = []
    
    for term in term_numbers:
        if term == 0:
            name = "MSE"
            file_name = f"iv_model_exact_loss_{name}_best.keras"
        elif term == 1:
            name = "Term1"
            file_name = f"iv_model_exact_loss_{name}_best.keras"
        elif term == 12:
            name = "Term1+2"
            file_name = f"iv_model_exact_loss_{name}_best.keras"
        elif term == 123:
            name = "Term1+2+3"
            file_name = f"iv_model_exact_loss_{name}_best.keras"
        elif term == 1234:
            name = "Full"
            file_name = f"iv_model_exact_loss_{name}_best.keras"
        else:
            continue
        
        # Add to lists
        path = os.path.join('models', file_name)
        if os.path.exists(path):
            model_paths.append(path)
            model_names.append(name)
        else:
            print(f"Warning: Model file {path} not found. Skipping.")
    
    # Load models
    models = load_models(model_paths, model_names)
    
    if not models:
        print("Error: No models could be loaded.")
        return
    
    # Create comparison plots
    create_comparison_plots(models, original_data, output_dir, preprocessed_data)
    
    print(f"Term comparison complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compare exact custom loss models with different term combinations')
    parser.add_argument('--models', nargs='+', required=False,
                       help='Paths to model files to compare')
    parser.add_argument('--names', nargs='+', required=False,
                       help='Names for each model in the plots')
    parser.add_argument('--terms', nargs='+', type=int, required=False,
                       help='Term combinations to compare (0=MSE, 1=Term1, 12=Term1+2, 123=Term1+2+3, 1234=Full)')
    parser.add_argument('--data', type=str, default='iv_data.csv', 
                       help='Path to original I-V data CSV file')
    parser.add_argument('--output_dir', type=str, default='exact_loss_comparison',
                       help='Directory to save comparison plots')
    parser.add_argument('--preprocessed_data', type=str, default='preprocessed_data.pkl',
                       help='Path to preprocessed data pickle file')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found")
        return
    
    # If term numbers are provided, use that approach
    if args.terms:
        compare_term_models(args.terms, args.data, args.preprocessed_data, args.output_dir)
        return
    
    # If specific models are provided
    if args.models and args.names:
        if len(args.models) != len(args.names):
            print("Error: Number of models must match number of names")
            return
        
        # Load original data
        original_data = load_original_data(args.data)
        
        # Load preprocessed data if available
        preprocessed_data = None
        if os.path.exists(args.preprocessed_data):
            preprocessed_data = load_preprocessed_data(args.preprocessed_data)
        
        # Load models
        models = load_models(args.models, args.names)
        
        if not models:
            print("Error: No models loaded.")
            return
        
        # Create comparison plots
        create_comparison_plots(models, original_data, args.output_dir, preprocessed_data)
    else:
        print("Error: Either --terms or both --models and --names must be provided")
        return

if __name__ == "__main__":
    main()