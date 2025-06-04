"""
Main script to run the exact custom loss implementation from the paper.
This script will first preprocess the data, then train the model with the exact loss.
"""
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run exact custom loss implementation')
    parser.add_argument('--data_file', type=str, default='iv_data.csv', 
                        help='CSV file containing I-V data')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    
    args = parser.parse_args()
    
    # Check if the data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found.")
        return
    
    # Step 1: Preprocess data (with derivatives)
    print("Step 1: Preprocessing data with derivative calculations...")
    preprocess_cmd = f"python run_data_loader.py --file {args.data_file}"
    result = subprocess.run(preprocess_cmd, shell=True, check=True)
    
    if result.returncode != 0:
        print("Error: Data preprocessing failed.")
        return
    
    # Step 2: Train model with exact custom loss
    print("\nStep 2: Training model with exact custom loss...")
    train_cmd = f"python train_with_exact_loss.py --epochs {args.epochs} --batch_size {args.batch_size}"
    result = subprocess.run(train_cmd, shell=True, check=True)
    
    if result.returncode != 0:
        print("Error: Model training failed.")
        return
    
    print("\nComplete! The model has been trained with the exact custom loss from the paper.")
    print("Model file: models/iv_model_exact_loss_final.h5")
    print("Training history: plots/history_exact_loss.png")
    print("Evaluation plots: evaluation/id_vg_Exact_Loss.png and evaluation/id_vd_Exact_Loss.png")

if __name__ == "__main__":
    main()