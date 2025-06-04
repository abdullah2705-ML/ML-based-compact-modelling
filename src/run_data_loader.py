from data_loader import DataLoader
import os
import argparse

def main():
    """Run the data preprocessing script"""
    parser = argparse.ArgumentParser(description='Preprocess I-V data for neural network modeling')
    parser.add_argument('--file', type=str, default='iv_data.csv', help='Path to the CSV file containing I-V data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use as test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        print("Please provide a valid CSV file with columns 'Vg', 'Vd', and 'Id'.")
        return
    
    print(f"Loading and preprocessing data from {args.file}...")
    
    # Create data loader and process data
    loader = DataLoader(
        file_path=args.file,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Load and process the data
    data = loader.load_data()
    
    print("Data preprocessing completed successfully.")
    print(f"Total samples: {len(data['X_train']) + len(data['X_test'])}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Testing samples: {len(data['X_test'])}")
    print(f"Data saved to 'preprocessed_data.pkl' and CSV files.")

if __name__ == "__main__":
    main()