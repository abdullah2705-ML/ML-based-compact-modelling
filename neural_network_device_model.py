# neural_network_device_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import time

# Enable mixed precision for faster inference
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class NNDeviceModel:
    """Neural network-based device model for I-V characteristics"""
    
    def __init__(self):
        self.iv_model = None
        # Add JIT compilation for better performance
        self.iv_predict_function = None
    
    def _create_iv_model(self, hidden_layers=[8, 8, 8]):
        """Create I-V neural network model structure"""
        model = models.Sequential()
        model.add(layers.Input(shape=(2,)))  # VGS, VDS inputs
        
        # Add hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='tanh'))
        
        # Output layer (no activation - we'll apply custom transfer function)
        model.add(layers.Dense(1))
        
        return model
    
    def _iv_output_transfer_function(self, y, vds):
        """Apply I-V output transfer function: I_DS = V_DS * e^y"""
        return vds * tf.exp(y)
    
    def load_csv_data(self, filepath):
        """Load and preprocess I-V data from CSV file"""
        # Load data from CSV
        data = pd.read_csv(filepath)
        
        # Extract columns
        vgs = data['Vg'].values
        vds = data['Vd'].values
        ids = data['Id'].values
        
        # Create feature matrix X and target y
        X = np.column_stack([vgs, vds])
        
        # Filter out points where VDS = 0 to avoid division by zero in log transform
        non_zero_vds = vds != 0
        X_filtered = X[non_zero_vds]
        ids_filtered = ids[non_zero_vds]
        vds_filtered = vds[non_zero_vds]
        
        # Calculate y = ln(IDS/VDS) for transfer function
        y = np.log(np.abs(ids_filtered / vds_filtered))
        
        # Create a complete dataset with relevant values
        dataset = np.column_stack([X_filtered, ids_filtered, y, vds_filtered])
        
        print(f"Loaded {len(dataset)} data points from CSV (after filtering VDS=0 points)")
        return dataset
        
    def train_iv_model(self, dataset, hidden_layers=[8, 8, 8], epochs=1000, batch_size=128):
        """Train the I-V neural network model"""
        # Create and compile model
        self.iv_model = self._create_iv_model(hidden_layers)
        
        # Extract data
        X = dataset[:, :2]  # VGS, VDS
        y = dataset[:, 3]   # ln(IDS/VDS)
        vds = dataset[:, 4:5] if dataset.shape[1] > 4 else dataset[:, 1:2]  # VDS
        
        # Train-test split
        X_train, X_test, y_train, y_test, vds_train, vds_test = train_test_split(
            X, y, vds, test_size=0.2, random_state=42)
        
        # Reshape y for model training
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Custom training loop with TensorFlow
        self.iv_model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
        history = self.iv_model.fit(X_train, y_train, 
                                    validation_data=(X_test, y_test), 
                                    epochs=epochs, 
                                    batch_size=batch_size)
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        self.iv_model.save('models/iv_model.keras')
        
        # Create a JIT-compiled function for faster inference
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 2], dtype=tf.float32)
        ])
        def predict_ids_batch(inputs):
            y_pred = self.iv_model(inputs, training=False)
            vds = tf.cast(inputs[:, 1:2], y_pred.dtype)  # Cast to match y_pred dtype
            ids = vds * tf.exp(y_pred)
            return ids
        
        self.iv_predict_function = predict_ids_batch
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/training_history.png', dpi=300)
        plt.close()
        
        return self.iv_model
    
    def predict_ids(self, vgs, vds):
        """Predict drain-source current"""
        if isinstance(vgs, (list, np.ndarray)) and isinstance(vds, (list, np.ndarray)):
            # Batch processing
            X = np.column_stack([vgs, vds])
            X = X.astype(np.float32)
            
            # Use JIT-compiled function if available, otherwise use model directly
            if self.iv_predict_function is not None:
                ids = self.iv_predict_function(X).numpy()
            else:
                # Get prediction from model
                y_pred = self.iv_model(X, training=False)
                
                # Convert vds to the same dtype as y_pred for multiplication
                vds_tensor = tf.convert_to_tensor(X[:, 1:2], dtype=y_pred.dtype)
                
                # Apply output transfer function
                ids = vds_tensor * tf.exp(y_pred)
                ids = ids.numpy()
            
            return ids.flatten()
        else:
            # Single point processing
            X = np.array([[vgs, vds]], dtype=np.float32)
            
            # Get prediction from model
            y_pred = self.iv_model(X, training=False)
            
            # Apply output transfer function
            ids = vds * tf.exp(y_pred)
            
            return ids.numpy()[0][0]
    
    def compare_speed_with_hspice(self, num_points=1000000):
        """Compare inference speed with simulated HSPICE/actual data model"""
        # Generate random test points
        np.random.seed(42)
        test_points = np.random.uniform(0, 0.65, (num_points, 2))
        
        # Filter out points with vds=0 to avoid division by zero
        valid_indices = test_points[:, 1] != 0
        test_points = test_points[valid_indices]
        
        # Extract vgs and vds arrays
        vgs_array = test_points[:, 0]
        vds_array = test_points[:, 1]
        
        # Measure NN model inference time (batch processing)
        start_time = time.time()
        _ = self.predict_ids(vgs_array, vds_array)
        nn_time = time.time() - start_time
        
        # We don't have the HSPICE model here, so we'll just report our time
        print(f"NN model inference time for {len(vgs_array)} points: {nn_time:.4f} seconds")
        print(f"Average time per point: {nn_time/len(vgs_array)*1e6:.2f} microseconds")
        
        return nn_time

    def plot_iv_characteristics(self, dataset):
        """Plot I-V characteristics using actual data vs model predictions"""
        os.makedirs('plots', exist_ok=True)
        
        # Extract unique VGS and VDS values from the dataset
        X = dataset[:, :2]
        vgs_values = sorted(list(set(X[:, 0])))
        vds_values = sorted(list(set(X[:, 1])))
        
        # Filter out VDS=0 for transfer characteristics
        vds_values = [v for v in vds_values if v != 0]
        
        # Plot Id-Vg characteristics (transfer)
        plt.figure(figsize=(10, 8))
        selected_vds = [vds_values[0], vds_values[len(vds_values)//2], vds_values[-1]]
        
        for vds in selected_vds:
            # Filter data points for this VDS
            mask = np.isclose(X[:, 1], vds)
            vgs_data = X[mask, 0]
            ids_data = dataset[mask, 2]  # Original IDS data
            
            # Sort by VGS for plotting
            sort_idx = np.argsort(vgs_data)
            vgs_data = vgs_data[sort_idx]
            ids_data = ids_data[sort_idx]
            
            # Generate predictions
            ids_pred = self.predict_ids(vgs_data, np.full_like(vgs_data, vds))
            
            # Plot data and predictions
            plt.semilogy(vgs_data, np.abs(ids_data), 'o', markersize=3, label=f'Data, Vds={vds:.2f}V')
            plt.semilogy(vgs_data, np.abs(ids_pred), '-', label=f'NN, Vds={vds:.2f}V')
        
        plt.xlabel('Vgs (V)')
        plt.ylabel('|Ids| (A)')
        plt.title('Transfer Characteristics (Id-Vg)')
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/id_vg_characteristics.png', dpi=300)
        plt.close()
        
        # Plot Id-Vd characteristics (output)
        plt.figure(figsize=(10, 8))
        vgs_values = np.array(vgs_values)
        # Select a few representative VGS values
        if len(vgs_values) > 4:
            selected_vgs = [vgs_values[len(vgs_values)//5], 
                           vgs_values[2*len(vgs_values)//5],
                           vgs_values[3*len(vgs_values)//5],
                           vgs_values[4*len(vgs_values)//5]]
        else:
            selected_vgs = vgs_values
        
        for vgs in selected_vgs:
            # Filter data points for this VGS
            mask = np.isclose(X[:, 0], vgs)
            vds_data = X[mask, 1]
            ids_data = dataset[mask, 2]  # Original IDS data
            
            # Remove VDS=0 points
            non_zero = vds_data != 0
            vds_data = vds_data[non_zero]
            ids_data = ids_data[non_zero]
            
            # Sort by VDS for plotting
            sort_idx = np.argsort(vds_data)
            vds_data = vds_data[sort_idx]
            ids_data = ids_data[sort_idx]
            
            # Generate predictions
            ids_pred = self.predict_ids(np.full_like(vds_data, vgs), vds_data)
            
            # Plot data and predictions
            plt.plot(vds_data, ids_data, 'o', markersize=3, label=f'Data, Vgs={vgs:.2f}V')
            plt.plot(vds_data, ids_pred, '-', label=f'NN, Vgs={vgs:.2f}V')
        
        plt.xlabel('Vds (V)')
        plt.ylabel('Ids (A)')
        plt.title('Output Characteristics (Id-Vd)')
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/id_vd_characteristics.png', dpi=300)
        plt.close()