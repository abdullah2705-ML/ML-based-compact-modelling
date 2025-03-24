# neural_network_device_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import time

# Enable mixed precision for faster inference
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class DeviceDataGenerator:
    """Generates synthetic data that mimics transistor I-V and C-V characteristics"""
    
    def __init__(self, vgs_range=(-0.65, 0.65), vds_range=(-0.65, 0.65), step=0.01):
        self.vgs_min, self.vgs_max = vgs_range
        self.vds_min, self.vds_max = vds_range
        self.step = step
        
        # Create voltage ranges (exclude vds=0 to avoid division by zero)
        self.vgs_values = np.arange(self.vgs_min, self.vgs_max + self.step, self.step)
        self.vds_values = np.arange(self.vds_min, self.vds_max + self.step, self.step)
        self.vds_values = self.vds_values[self.vds_values != 0]
        
        # Constants for our synthetic model
        self.vt = 0.2  # Threshold voltage
        self.mu = 0.01  # Mobility
        self.cox = 2e-2  # Oxide capacitance
        self.w = 20e-9  # Width
        self.l = 12e-9  # Length
        self.lambda_param = 0.1  # Channel length modulation
        
    def _calculate_ids(self, vgs, vds):
        """Calculate drain-source current using simplified transistor model"""
        # Vectorized implementation for batch processing
        if isinstance(vgs, np.ndarray) and isinstance(vds, np.ndarray) and vgs.shape == vds.shape:
            # Vectorized batch processing
            vt_effective = self.vt + 0.03 * vds  # DIBL effect
            
            # Initialize output array
            ids = np.zeros_like(vgs)
            
            # Subthreshold region
            subthresh_mask = vgs <= vt_effective
            ids[subthresh_mask] = 1e-9 * np.exp((vgs[subthresh_mask] - vt_effective[subthresh_mask]) / 0.06) * \
                                (1 - np.exp(-vds[subthresh_mask] / 0.026))
            
            # Above threshold
            above_mask = ~subthresh_mask
            # Linear region
            linear_mask = above_mask & (np.abs(vds) < (vgs - vt_effective))
            ids[linear_mask] = self.mu * self.cox * (self.w / self.l) * \
                            ((vgs[linear_mask] - vt_effective[linear_mask]) * vds[linear_mask] - 0.5 * vds[linear_mask]**2)
            
            # Saturation region
            sat_mask = above_mask & ~linear_mask
            ids[sat_mask] = 0.5 * self.mu * self.cox * (self.w / self.l) * \
                          (vgs[sat_mask] - vt_effective[sat_mask])**2 * \
                          (1 + self.lambda_param * vds[sat_mask])
            
            # Ensure ids has the same sign as vds
            return ids * np.sign(vds)
        else:
            # Single point calculation (keeping original implementation for compatibility)
            vt_effective = self.vt + 0.03 * vds  # DIBL effect
            
            if vgs <= vt_effective:  # Subthreshold
                ids = 1e-9 * np.exp((vgs - vt_effective) / 0.06) * (1 - np.exp(-vds / 0.026))
            else:  # Above threshold
                if abs(vds) < (vgs - vt_effective):  # Linear region
                    ids = self.mu * self.cox * (self.w / self.l) * ((vgs - vt_effective) * vds - 0.5 * vds**2)
                else:  # Saturation region
                    ids = 0.5 * self.mu * self.cox * (self.w / self.l) * (vgs - vt_effective)**2 * (1 + self.lambda_param * vds)
            
            # Ensure ids has the same sign as vds
            return ids * np.sign(vds)
    
    def _calculate_qg(self, vgs, vds):
        """Calculate gate charge using simplified model"""
        # Vectorized implementation for batch processing
        if isinstance(vgs, np.ndarray) and isinstance(vds, np.ndarray) and vgs.shape == vds.shape:
            vt_effective = self.vt + 0.02 * vds  # DIBL effect on threshold
            
            # Initialize output array
            qg = np.zeros_like(vgs)
            
            # Subthreshold region
            subthresh_mask = vgs <= vt_effective
            qg[subthresh_mask] = self.cox * self.w * self.l * 0.1 * (vgs[subthresh_mask] - vt_effective[subthresh_mask])
            
            # Above threshold
            above_mask = ~subthresh_mask
            qg[above_mask] = self.cox * self.w * self.l * (vgs[above_mask] - vt_effective[above_mask])
            
            # Add non-linearity and vds dependency
            qg += 0.1 * self.cox * self.w * self.l * vds * np.tanh(vgs)
            
            return qg
        else:
            # Single point calculation
            vt_effective = self.vt + 0.02 * vds  # DIBL effect on threshold
            
            if vgs <= vt_effective:
                qg = self.cox * self.w * self.l * 0.1 * (vgs - vt_effective)  # Small charge in subthreshold
            else:
                qg = self.cox * self.w * self.l * (vgs - vt_effective)  # Above threshold
            
            # Add some non-linearity and vds dependency
            qg += 0.1 * self.cox * self.w * self.l * vds * np.tanh(vgs)
            
            return qg
    
    def generate_iv_data(self):
        """Generate I-V data points with vectorized operations"""
        # Create a mesh grid of all VGS and VDS combinations
        vgs_mesh, vds_mesh = np.meshgrid(self.vgs_values, self.vds_values)
        vgs_flat = vgs_mesh.flatten()
        vds_flat = vds_mesh.flatten()
        
        # Calculate IDS for all points at once
        ids_flat = self._calculate_ids(vgs_flat, vds_flat)
        
        # Calculate derivatives for training purposes
        delta = 1e-4
        
        # gm - transconductance (vectorized)
        vgs_delta = vgs_flat + delta
        ids_dvgs = self._calculate_ids(vgs_delta, vds_flat)
        gm = (ids_dvgs - ids_flat) / delta
        
        # gd - output conductance (vectorized)
        vds_delta = vds_flat + delta
        ids_dvds = self._calculate_ids(vgs_flat, vds_delta)
        gd = (ids_dvds - ids_flat) / delta
        
        # Stack all data into a single array
        data_points = np.column_stack([vgs_flat, vds_flat, ids_flat, gm, gd])
        
        return data_points
    
    def generate_cv_data(self):
        """Generate C-V data points with vectorized operations"""
        # Create a mesh grid of all VGS and VDS combinations
        vgs_mesh, vds_mesh = np.meshgrid(self.vgs_values, self.vds_values)
        vgs_flat = vgs_mesh.flatten()
        vds_flat = vds_mesh.flatten()
        
        # Calculate QG for all points at once
        qg_flat = self._calculate_qg(vgs_flat, vds_flat)
        
        # Calculate derivatives for capacitances (vectorized)
        delta = 1e-4
        
        # Cgg - gate-gate capacitance
        vgs_delta = vgs_flat + delta
        qg_dvgs = self._calculate_qg(vgs_delta, vds_flat)
        cgg = (qg_dvgs - qg_flat) / delta
        
        # Cgd - gate-drain capacitance
        vds_delta = vds_flat + delta
        qg_dvds = self._calculate_qg(vgs_flat, vds_delta)
        cgd = (qg_dvds - qg_flat) / delta
        
        # Stack all data into a single array
        data_points = np.column_stack([vgs_flat, vds_flat, qg_flat, cgg, cgd])
        
        return data_points

    def save_data(self, output_dir):
        """Generate and save both I-V and C-V data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save I-V data
        iv_data = self.generate_iv_data()
        np.save(os.path.join(output_dir, 'iv_data.npy'), iv_data)
        
        # Generate and save C-V data
        cv_data = self.generate_cv_data()
        np.save(os.path.join(output_dir, 'cv_data.npy'), cv_data)
        
        print(f"Data saved to {output_dir}")
        return iv_data, cv_data

class NNDeviceModel:
    """Neural network-based device model for I-V and C-V characteristics"""
    
    def __init__(self):
        self.iv_model = None
        self.cv_model = None
        # Add JIT compilation for better performance
        self.iv_predict_function = None
        self.cv_predict_function = None
    
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
    
    def _create_cv_model(self, hidden_layers=[15, 10]):
        """Create C-V neural network model structure"""
        model = models.Sequential()
        model.add(layers.Input(shape=(2,)))  # VGS, VDS inputs
        
        # Add hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='tanh'))
        
        # Output layer (no activation - will be scaled later)
        model.add(layers.Dense(1))
        
        return model
    
    def _iv_output_transfer_function(self, y, vds):
        """Apply I-V output transfer function: I_DS = V_DS * e^y"""
        return vds * tf.exp(y)
    
    def _iv_custom_loss(self, y_true, y_pred, vds, vgs_tensor, a=1.0, b=1.0, c=500.0, d=70.0, delta=1e-6):
        """Custom loss function for I-V training as described in the paper"""
        # Extract true values
        ids_true = vds * tf.exp(y_true)
        
        # Output transfer derivatives
        z_pred = tf.exp(y_pred)
        dz_dvds_pred = tf.gradients(z_pred, vds)[0]
        
        # Calculate derivatives for true and predicted
        gm_true = tf.gradients(ids_true, vgs_tensor)[0]
        gd_true = tf.gradients(ids_true, vds)[0]
        
        # Apply output transfer function to predictions
        ids_pred = vds * tf.exp(y_pred)
        gm_pred = tf.gradients(ids_pred, vgs_tensor)[0]
        gd_pred = tf.gradients(ids_pred, vds)[0]
        
        # True derivatives for z
        z_true = tf.exp(y_true)
        dz_dvds_true = tf.gradients(z_true, vds)[0]
        
        # Compute loss components
        loss_y = tf.reduce_mean(tf.square((y_true - y_pred) / (y_true + delta)))
        loss_gd = tf.reduce_mean(tf.square((gd_true - gd_pred) / (gd_true + delta)))
        loss_gm = tf.reduce_mean(tf.square(gm_true - gm_pred))
        loss_dz = tf.reduce_mean(tf.square(dz_dvds_true - dz_dvds_pred))
        
        # Combine all loss components
        total_loss = a * loss_y + b * loss_gd + c * loss_gm + d * loss_dz
        
        return total_loss
    
    def _cv_custom_loss(self, y_true, y_pred, vgs_tensor, vds_tensor, a=1.0, b=12.0, c=10.0, d=1.3):
        """Custom loss function for C-V training as described in the paper"""
        # Calculate derivatives for true and predicted
        dy_dvgs_true = tf.gradients(y_true, vgs_tensor)[0]
        dy_dvds_true = tf.gradients(y_true, vds_tensor)[0]
        d2y_dvds2_true = tf.gradients(dy_dvds_true, vds_tensor)[0]
        
        dy_dvgs_pred = tf.gradients(y_pred, vgs_tensor)[0]
        dy_dvds_pred = tf.gradients(y_pred, vds_tensor)[0]
        d2y_dvds2_pred = tf.gradients(dy_dvds_pred, vds_tensor)[0]
        
        # Compute loss components
        loss_y = tf.reduce_mean(tf.square(y_true - y_pred))
        loss_dvgs = tf.reduce_mean(tf.square(dy_dvgs_true - dy_dvgs_pred))
        loss_dvds = tf.reduce_mean(tf.square(dy_dvds_true - dy_dvds_pred))
        loss_d2vds = tf.reduce_mean(tf.square(d2y_dvds2_true - d2y_dvds2_pred))
        
        # Combine all loss components
        total_loss = a * loss_y + b * loss_dvgs + c * loss_dvds + d * loss_d2vds
        
        return total_loss
        
    def train_iv_model(self, iv_data, hidden_layers=[8, 8, 8], epochs=100, batch_size=128):
        """Train the I-V neural network model"""
        # Create and compile model
        self.iv_model = self._create_iv_model(hidden_layers)
        
        # Extract data
        X = iv_data[:, :2]  # VGS, VDS
        vds = iv_data[:, 1:2]  # VDS
        ids = iv_data[:, 2:3]  # IDS
        
        # Calculate y = ln(IDS/VDS) for transfer function
        y = np.log(np.abs(ids / vds))
        
        # Train-test split
        X_train, X_test, y_train, y_test, vds_train, vds_test = train_test_split(
            X, y, vds, test_size=0.2, random_state=42)
        
        # Custom training loop with TensorFlow
        self.iv_model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
        self.iv_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                         epochs=epochs, batch_size=batch_size)
        
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
        
        return self.iv_model
    
    def train_cv_model(self, cv_data, hidden_layers=[15, 10], epochs=100, batch_size=128):
        """Train the C-V neural network model"""
        # Create and compile model
        self.cv_model = self._create_cv_model(hidden_layers)
        
        # Extract data
        X = cv_data[:, :2]  # VGS, VDS
        qg = cv_data[:, 2:3]  # QG
        
        # Scale charge data
        scaling_factor = 1e9  # Adjust based on the magnitude of your charge values
        y = qg * scaling_factor
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.cv_model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
        self.cv_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                         epochs=epochs, batch_size=batch_size)
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        self.cv_model.save('models/cv_model.keras')
        
        # Create a JIT-compiled function for faster inference
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 2], dtype=tf.float32)
        ])
        def predict_qg_batch(inputs):
            y_pred = self.cv_model(inputs, training=False)
            scaling_factor = 1e9  # Same as used during training
            qg = y_pred / scaling_factor
            return qg
        
        self.cv_predict_function = predict_qg_batch
        
        return self.cv_model
    
    def predict_ids(self, vgs, vds):
        """Predict drain-source current"""
        if isinstance(vgs, (list, np.ndarray)) and isinstance(vds, (list, np.ndarray)):
            # Batch processing
            X = np.column_stack([vgs, vds])
            X = X.astype(np.float32)
            
            # Get prediction from model
            y_pred = self.iv_model(X, training=False)
            
            # Convert vds to the same dtype as y_pred for multiplication
            vds_tensor = tf.convert_to_tensor(X[:, 1:2], dtype=y_pred.dtype)
            
            # Apply output transfer function
            ids = vds_tensor * tf.exp(y_pred)
            
            return ids.numpy().flatten()
        else:
            # Single point processing
            X = np.array([[vgs, vds]], dtype=np.float32)
            
            # Get prediction from model
            y_pred = self.iv_model(X, training=False)
            
            # Convert vds to the same dtype as y_pred for multiplication
            vds_tensor = tf.convert_to_tensor([[vds]], dtype=y_pred.dtype)
            
            # Apply output transfer function
            ids = vds_tensor * tf.exp(y_pred)
            
            return ids.numpy()[0][0]
    
    def predict_qg(self, vgs, vds):
        """Predict gate charge"""
        if isinstance(vgs, (list, np.ndarray)) and isinstance(vds, (list, np.ndarray)):
            # Batch processing
            X = np.column_stack([vgs, vds])
            X = X.astype(np.float32)
            y_pred = self.cv_predict_function(X).numpy()
            return y_pred.flatten()
        else:
            # Single point processing
            X = np.array([[vgs, vds]], dtype=np.float32)
            y_pred = self.cv_predict_function(X).numpy()
            return y_pred[0][0]
    
    def compare_speed_with_bsim(self, num_points=1000000):
        """Compare inference speed with simulated BSIM model"""
        # Generate random test points
        np.random.seed(42)
        test_points = np.random.uniform(-0.6, 0.6, (num_points, 2))
        
        # Filter out points with vds=0 to avoid division by zero
        valid_indices = test_points[:, 1] != 0
        test_points = test_points[valid_indices]
        
        # Create a simulated BSIM function (actually our data generator)
        data_gen = DeviceDataGenerator()
        
        # Extract vgs and vds arrays
        vgs_array = test_points[:, 0]
        vds_array = test_points[:, 1]
        
        # Measure NN model inference time (batch processing)
        start_time = time.time()
        _ = self.predict_ids(vgs_array, vds_array)
        nn_time = time.time() - start_time
        
        # Measure simulated BSIM time (vectorized)
        start_time = time.time()
        _ = data_gen._calculate_ids(vgs_array, vds_array)
        bsim_time = time.time() - start_time
        
        print(f"NN model inference time: {nn_time:.4f} seconds")
        print(f"Simulated BSIM time: {bsim_time:.4f} seconds")
        print(f"Speedup: {bsim_time/nn_time:.2f}x")
        
        return nn_time, bsim_time

    def plot_iv_characteristics(self, data_generator):
        """Plot I-V characteristics comparing model predictions with synthetic data"""
        os.makedirs('plots', exist_ok=True)
        
        # Plot Id-Vg characteristics
        plt.figure(figsize=(10, 8))
        vds_values = [0.05, 0.3, 0.6]
        
        for vds in vds_values:
            # Generate ground truth
            vgs_range = np.linspace(-0.6, 0.6, 100)
            
            # Vectorized calculation for both true and predicted values
            ids_true = data_generator._calculate_ids(vgs_range, np.full_like(vgs_range, vds))
            ids_pred = self.predict_ids(vgs_range, np.full_like(vgs_range, vds))
            
            plt.semilogy(vgs_range, np.abs(ids_true), '-', label=f'True, Vds={vds}V')
            plt.semilogy(vgs_range, np.abs(ids_pred), 'o', markersize=3, label=f'NN, Vds={vds}V')
        
        plt.xlabel('Vgs (V)')
        plt.ylabel('|Ids| (A)')
        plt.title('Transfer Characteristics (Id-Vg)')
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/id_vg_characteristics.png', dpi=300)
        
        # Plot Id-Vd characteristics
        plt.figure(figsize=(10, 8))
        vgs_values = [0.3, 0.4, 0.5, 0.6]
        
        for vgs in vgs_values:
            # Generate ground truth
            vds_range = np.linspace(0.01, 0.6, 100)  # Avoid vds=0
            
            # Vectorized calculation
            ids_true = data_generator._calculate_ids(np.full_like(vds_range, vgs), vds_range)
            ids_pred = self.predict_ids(np.full_like(vds_range, vgs), vds_range)
            
            plt.plot(vds_range, ids_true, '-', label=f'True, Vgs={vgs}V')
            plt.plot(vds_range, ids_pred, 'o', markersize=3, label=f'NN, Vgs={vgs}V')
        
        plt.xlabel('Vds (V)')
        plt.ylabel('Ids (A)')
        plt.title('Output Characteristics (Id-Vd)')
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/id_vd_characteristics.png', dpi=300)

    def plot_cv_characteristics(self, data_generator):
        """Plot C-V characteristics comparing model predictions with synthetic data"""
        os.makedirs('plots', exist_ok=True)
        
        # Plot Qg-Vg characteristics
        plt.figure(figsize=(10, 8))
        vds_values = [0.05, 0.3, 0.6]
        
        for vds in vds_values:
            # Generate ground truth
            vgs_range = np.linspace(-0.6, 0.6, 100)
            
            # Vectorized calculation
            qg_true = data_generator._calculate_qg(vgs_range, np.full_like(vgs_range, vds))
            qg_pred = self.predict_qg(vgs_range, np.full_like(vgs_range, vds))
            
            plt.plot(vgs_range, qg_true, '-', label=f'True, Vds={vds}V')
            plt.plot(vgs_range, qg_pred, 'o', markersize=3, label=f'NN, Vds={vds}V')
        
        plt.xlabel('Vgs (V)')
        plt.ylabel('Qg (C)')
        plt.title('Gate Charge vs Gate Voltage')
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/qg_vg_characteristics.png', dpi=300)
        
        # Calculate and plot Cgg (dQg/dVgs)
        plt.figure(figsize=(10, 8))
        vds_values = [0.05, 0.3, 0.6]
        
        for vds in vds_values:
            # Generate ground truth
            vgs_range = np.linspace(-0.55, 0.55, 100)  # Slightly smaller range for derivative calculation
            
            # Calculate numerical derivatives (vectorized)
            delta = 0.01
            vgs_minus = vgs_range - delta
            vgs_plus = vgs_range + delta
            
            qg1_true = data_generator._calculate_qg(vgs_minus, np.full_like(vgs_range, vds))
            qg2_true = data_generator._calculate_qg(vgs_plus, np.full_like(vgs_range, vds))
            cgg_true = (qg2_true - qg1_true) / (2 * delta)
            
            qg1_pred = self.predict_qg(vgs_minus, np.full_like(vgs_range, vds))
            qg2_pred = self.predict_qg(vgs_plus, np.full_like(vgs_range, vds))
            cgg_pred = (qg2_pred - qg1_pred) / (2 * delta)
            
            plt.plot(vgs_range, cgg_true, '-', label=f'True, Vds={vds}V')
            plt.plot(vgs_range, cgg_pred, 'o', markersize=3, label=f'NN, Vds={vds}V')
        
        plt.xlabel('Vgs (V)')
        plt.ylabel('Cgg (F)')
        plt.title('Gate-Gate Capacitance')
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/cgg_characteristics.png', dpi=300)