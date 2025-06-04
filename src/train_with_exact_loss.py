import os
import argparse
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from model_with_exact_loss import ExactLossModel

def train_with_exact_loss(term=1234, epochs=500, batch_size=32):
    """Train a model using the exact custom loss function from the paper"""
    # Load preprocessed data
    data_path = 'preprocessed_data.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return False
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded successfully.")
    
    # Create directories for outputs
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # Set TensorFlow memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
    
    # Get term name for display and file naming
    term_name_map = {
        0: "MSE",
        1: "Term1",
        12: "Term1+2",
        123: "Term1+2+3",
        1234: "Full"
    }
    term_name = term_name_map.get(term, f"Custom({term})")
    
    # Create a fresh model with specified term selection
    model = ExactLossModel(hidden_layers=(8, 8, 8), learning_rate=5e-4, term_selection=term)
    model.build_model()
    
    print(f"Training model with exact {term_name} loss...")
    
    # Train the model - added fresh_start=True to always start from scratch
    history = model.train(
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        fresh_start=True
    )
    
    model_path = f'models/iv_model_exact_loss_{term_name}_final.keras'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save training history
    plt.figure(figsize=(12, 8))
    
    # Plot overall loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {term_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Plot individual terms
    plt.subplot(2, 2, 2)
    for term_key in ['term1', 'term2', 'term3', 'term4']:
        plt.plot(history[term_key], label=term_key)
    plt.title('Loss Components')
    plt.ylabel('Loss Value')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    history_path = f'plots/history_exact_loss_{term_name}.png'
    plt.tight_layout()
    plt.savefig(history_path, dpi=300)
    plt.close()
    print(f"Training history saved to {history_path}")
    
    # Evaluate the model
    evaluate_model(model, data, term_name)
    
    return model, history

def evaluate_model(model, data, model_name):
    """Evaluate model performance and generate plots"""
    # Extract test data
    X_test = data['X_test']
    y_test = data['y_test']
    X_test_orig = data['X_test_orig']
    id_test = data['id_test']
    vd_test = X_test_orig[:, 1]
    
    # Predict on test data
    y_pred = model.predict(X_test).numpy().flatten()
    id_pred = vd_test * np.exp(y_pred)
    
    # Calculate error metrics
    mape = np.mean(np.abs((id_test - id_pred) / id_test)) * 100
    rmse = np.sqrt(np.mean((id_test - id_pred) ** 2))
    
    print(f"Model Evaluation with {model_name} Loss:")
    print(f"MAPE: {mape:.4f}%")
    print(f"RMSE: {rmse:.6e}")
    
    # Generate Id-Vg plot for a specific Vd
    plot_id_vg_curves(X_test_orig, id_test, id_pred, model_name, 'evaluation')
    
    # Generate Id-Vd plot for a specific Vg
    plot_id_vd_curves(X_test_orig, id_test, id_pred, model_name, 'evaluation')
    
    return {
        'mape': mape,
        'rmse': rmse
    }

def plot_id_vg_curves(X_test_orig, id_test, id_pred, model_name, save_dir):
    """Plot Id-Vg curves for selected Vd values"""
    plt.figure(figsize=(10, 6))
    
    # Find unique Vd values
    vd_values = np.unique(X_test_orig[:, 1])
    
    # Select a Vd value to plot (middle of range)
    vd_target = vd_values[len(vd_values)//2]
    
    # Get data for this Vd
    mask = np.isclose(X_test_orig[:, 1], vd_target)
    vg_values = X_test_orig[mask, 0]
    id_true = id_test[mask]
    id_predictions = id_pred[mask]
    
    # Sort by Vg for smooth plotting
    sort_idx = np.argsort(vg_values)
    vg_values = vg_values[sort_idx]
    id_true = id_true[sort_idx]
    id_predictions = id_predictions[sort_idx]
    
    # Plot
    plt.plot(vg_values, id_true, 'o', label='True data', markersize=4)
    plt.plot(vg_values, id_predictions, '-r', label=f'{model_name} model', linewidth=2)
    
    plt.title(f'Id-Vg Characteristics (Vd={vd_target:.2f}V)')
    plt.xlabel('Gate Voltage (V)')
    plt.ylabel('Drain Current (A)')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'id_vg_{model_name}.png'), dpi=300)
    plt.close()

def plot_id_vd_curves(X_test_orig, id_test, id_pred, model_name, save_dir):
    """Plot Id-Vd curves for selected Vg values"""
    plt.figure(figsize=(10, 6))
    
    # Find unique Vg values
    vg_values = np.unique(X_test_orig[:, 0])
    
    # Select a Vg value to plot (middle of range)
    vg_target = vg_values[len(vg_values)//2]
    
    # Get data for this Vg
    mask = np.isclose(X_test_orig[:, 0], vg_target)
    vd_values = X_test_orig[mask, 1]
    id_true = id_test[mask]
    id_predictions = id_pred[mask]
    
    # Sort by Vd for smooth plotting
    sort_idx = np.argsort(vd_values)
    vd_values = vd_values[sort_idx]
    id_true = id_true[sort_idx]
    id_predictions = id_predictions[sort_idx]
    
    # Plot
    plt.plot(vd_values, id_true, 'o', label='True data', markersize=4)
    plt.plot(vd_values, id_predictions, '-r', label=f'{model_name} model', linewidth=2)
    
    plt.title(f'Id-Vd Characteristics (Vg={vg_target:.2f}V)')
    plt.xlabel('Drain Voltage (V)')
    plt.ylabel('Drain Current (A)')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'id_vd_{model_name}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train neural network model with exact custom loss from the paper')
    parser.add_argument('--term', type=int, default=1234, 
                       help='Loss term to use: 0=MSE, 1=Term1, 12=Term1+2, 123=Term1+2+3, 1234=All terms')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    
    args = parser.parse_args()
    
    train_with_exact_loss(term=args.term, epochs=args.epochs, batch_size=args.batch_size)



    # python train_with_exact_loss.py --term 123 --epochs 500 --batch_size 32