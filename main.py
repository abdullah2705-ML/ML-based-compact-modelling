# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from neural_network_device_model import DeviceDataGenerator, NNDeviceModel
import tensorflow as tf

def main():
    """Main function to demonstrate the neural network device modeling framework"""
    # Set TensorFlow configuration for better performance
    tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
    
    # Create directory structure
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("1. Generating synthetic device data...")
    # Generate synthetic data that mimics TCAD/BSIM data
    data_generator = DeviceDataGenerator(vgs_range=(-0.65, 0.65), vds_range=(-0.65, 0.65), step=0.02)
    iv_data, cv_data = data_generator.save_data('data')
    
    print("\n2. Training neural network models...")
    # Create and train the neural network models
    nn_model = NNDeviceModel()
    
    print("   Training I-V model...")
    nn_model.train_iv_model(iv_data, hidden_layers=[8, 8, 8], epochs=50, batch_size=64)
    
    print("   Training C-V model...")
    nn_model.train_cv_model(cv_data, hidden_layers=[15, 10], epochs=50, batch_size=64)
    
    print("\n3. Evaluating model performance...")
    # Plot I-V and C-V characteristics
    print("   Plotting I-V characteristics...")
    nn_model.plot_iv_characteristics(data_generator)
    
    print("   Plotting C-V characteristics...")
    nn_model.plot_cv_characteristics(data_generator)
    
    print("\n4. Comparing inference speed...")
    # Compare inference speed with simulated BSIM (use smaller number for testing)
    nn_model.compare_speed_with_bsim(num_points=100000)  # Reduced points for faster demonstration
    
    print("\nNeural network device modeling completed successfully!")

if __name__ == "__main__":
    main()