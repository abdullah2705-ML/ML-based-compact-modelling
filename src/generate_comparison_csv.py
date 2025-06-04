import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
import argparse
from model_with_exact_loss import ExactLossModel

def generate_comparison_csv(model_term=1234, output_file='iv_comparison.csv'):
    """
    Generate a CSV file comparing actual vs predicted drain current values
    
    Args:
        model_term: The term selection used for the model (0=MSE, 1=Term1, 12=Term1+2, etc.)
        output_file: Path to save the output CSV file
    """
    # Get term name for loading the correct model
    term_name_map = {
        0: "MSE",
        1: "Term1",
        12: "Term1+2",
        123: "Term1+2+3",
        1234: "Full"
    }
    term_name = term_name_map.get(model_term, f"Custom({model_term})")
    
    print(f"Loading data and model with {term_name} loss...")
    
    # Load preprocessed data
    data_path = 'preprocessed_data.pkl'
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return False
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Try to load both best and final models, prioritize best
    model_paths = [
        f'models/iv_model_exact_loss_{term_name}_final.keras'
    ]
    
    # Create model with same parameters as in training
    model = ExactLossModel(hidden_layers=(8, 8, 8), learning_rate=5e-4, term_selection=model_term)
    model.build_model()
    
    # Try to load model
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            if model.load_model(model_path):
                model_loaded = True
                break
    
    if not model_loaded:
        print("Error: Could not find or load any model!")
        print(f"Please train the model first using train_with_exact_loss.py with --term {model_term}")
        return False
    
    # Get original inputs and actual current values
    # For a full comparison, we'll use both training and test data
    X_train_orig = data['X_train_orig']
    id_train = data['id_train']
    X_test_orig = data['X_test_orig']
    id_test = data['id_test']
    
    # Combine both sets for a full comparison
    X_all_orig = np.vstack((X_train_orig, X_test_orig))
    id_all_actual = np.concatenate((id_train, id_test))
    
    # Get the scaled inputs for prediction
    x_scaler = data['x_scaler']
    X_all_scaled = x_scaler.transform(X_all_orig)
    
    print(f"Making predictions for {len(X_all_scaled)} data points...")
    
    # Perform prediction
    y_pred = model.predict(X_all_scaled).numpy().flatten()
    
    # Calculate Id from predicted y values
    # Remember: y = ln(Id/Vd), so Id = Vd * exp(y)
    vd_values = X_all_orig[:, 1]
    id_all_predicted = vd_values * np.exp(y_pred)
    
    # Create DataFrame for results
    results_df = pd.DataFrame({
        'Vg': X_all_orig[:, 0],
        'Vd': X_all_orig[:, 1],
        'Id_actual': id_all_actual,
        'Id_predicted': id_all_predicted
    })
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    # Calculate error metrics
    mape = np.mean(np.abs((id_all_actual - id_all_predicted) / id_all_actual)) * 100
    rmse = np.sqrt(np.mean((id_all_actual - id_all_predicted) ** 2))
    
    print(f"Results saved to {output_file}")
    print(f"Total data points: {len(results_df)}")
    print(f"Overall MAPE: {mape:.4f}%")
    print(f"Overall RMSE: {rmse:.6e}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comparison CSV for actual vs predicted drain current')
    parser.add_argument('--term', type=int, default=1234, 
                      help='Model term to use: 0=MSE, 1=Term1, 12=Term1+2, 123=Term1+2+3, 1234=All terms')
    parser.add_argument('--output', type=str, default='iv_comparison.csv',
                      help='Output CSV filename')
    
    args = parser.parse_args()
    
    generate_comparison_csv(model_term=args.term, output_file=args.output)