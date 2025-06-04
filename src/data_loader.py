import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DataLoader:
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.x_scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def load_data(self):
        """Load data from CSV file and calculate derivatives for exact loss"""
        print(f"Loading and preprocessing data from {self.file_path}...")
        df = pd.read_csv(self.file_path)
        
        # Extract columns
        vg = df['Vg'].values
        vd = df['Vd'].values
        id_values = df['Id'].values
        
        # Prepare input features (Vg, Vd)
        X = np.column_stack((vg, vd))
        
        # Transform input data
        X_scaled = self.x_scaler.fit_transform(X)
        
        # Prepare output - using the transformation from the paper
        # y = ln(Id/Vd) as per equation (1) in the paper
        y = np.log(id_values / vd)
        
        # Calculate derivatives needed for exact custom loss
        gd_values, gm_values, dz_dvds_values = self.calculate_derivatives(vg, vd, id_values)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Also keep the original inputs and outputs for evaluation
        X_train_orig, X_test_orig, id_train, id_test = train_test_split(
            X, id_values, test_size=self.test_size, random_state=self.random_state
        )
        
        # Split derivatives into train and test sets
        gd_train, gd_test = train_test_split(
            gd_values, test_size=self.test_size, random_state=self.random_state
        )
        gm_train, gm_test = train_test_split(
            gm_values, test_size=self.test_size, random_state=self.random_state
        )
        dz_dvds_train, dz_dvds_test = train_test_split(
            dz_dvds_values, test_size=self.test_size, random_state=self.random_state
        )
        
        # Save data to CSV and pickle format
        self.save_to_csv(X_train, X_test, y_train, y_test, X_train_orig, X_test_orig, id_train, id_test)
        self.save_to_pickle(
            X_train, X_test, y_train, y_test, X_train_orig, X_test_orig, id_train, id_test,
            gd_train, gd_test, gm_train, gm_test, dz_dvds_train, dz_dvds_test
        )
        
        print("Data preprocessing completed successfully.")
        print(f"Total samples: {len(X_train) + len(X_test)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_orig': X_train_orig,
            'X_test_orig': X_test_orig,
            'id_train': id_train,
            'id_test': id_test,
            'x_scaler': self.x_scaler,
            'gd_train': gd_train,
            'gd_test': gd_test,
            'gm_train': gm_train,
            'gm_test': gm_test,
            'dz_dvds_train': dz_dvds_train,
            'dz_dvds_test': dz_dvds_test
        }
    
    def calculate_derivatives(self, vg, vd, id_values):
        """Calculate exact derivatives needed for the custom loss function"""
        print("Calculating exact derivatives for custom loss...")
        
        # Create arrays of unique VG and VD values
        unique_vg = np.sort(np.unique(vg))
        unique_vd = np.sort(np.unique(vd))
        
        # Create a 2D grid for ID values
        id_grid = np.zeros((len(unique_vd), len(unique_vg)))
        
        # Fill the grid with ID values
        for i, (vg_val, vd_val, id_val) in enumerate(zip(vg, vd, id_values)):
            vg_idx = np.where(unique_vg == vg_val)[0][0]
            vd_idx = np.where(unique_vd == vd_val)[0][0]
            id_grid[vd_idx, vg_idx] = id_val
        
        # Calculate derivatives using central differences
        # gd = ∂Id/∂VDS
        gd_grid = np.zeros_like(id_grid)
        # gm = ∂Id/∂VGS
        gm_grid = np.zeros_like(id_grid)
        # z = Id/Vd
        z_grid = np.zeros_like(id_grid)
        # dz_dvds = ∂(Id/Vd)/∂VDS
        dz_dvds_grid = np.zeros_like(id_grid)
        
        # Calculate z = Id/Vd first
        for vd_idx, vd_val in enumerate(unique_vd):
            if vd_val != 0:  # Avoid division by zero
                z_grid[vd_idx, :] = id_grid[vd_idx, :] / vd_val
        
        # Calculate derivatives using gradients
        for vd_idx in range(len(unique_vd)):
            for vg_idx in range(len(unique_vg)):
                # Calculate gm = ∂Id/∂VGS
                if vg_idx == 0:
                    # Forward difference at left edge
                    gm_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx+1] - id_grid[vd_idx, vg_idx]) / (unique_vg[vg_idx+1] - unique_vg[vg_idx])
                elif vg_idx == len(unique_vg) - 1:
                    # Backward difference at right edge
                    gm_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx] - id_grid[vd_idx, vg_idx-1]) / (unique_vg[vg_idx] - unique_vg[vg_idx-1])
                else:
                    # Central difference for interior points
                    gm_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx+1] - id_grid[vd_idx, vg_idx-1]) / (unique_vg[vg_idx+1] - unique_vg[vg_idx-1])
                
                # Calculate gd = ∂Id/∂VDS
                if vd_idx == 0:
                    # Forward difference at bottom edge
                    gd_grid[vd_idx, vg_idx] = (id_grid[vd_idx+1, vg_idx] - id_grid[vd_idx, vg_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx])
                elif vd_idx == len(unique_vd) - 1:
                    # Backward difference at top edge
                    gd_grid[vd_idx, vg_idx] = (id_grid[vd_idx, vg_idx] - id_grid[vd_idx-1, vg_idx]) / (unique_vd[vd_idx] - unique_vd[vd_idx-1])
                else:
                    # Central difference for interior points
                    gd_grid[vd_idx, vg_idx] = (id_grid[vd_idx+1, vg_idx] - id_grid[vd_idx-1, vg_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx-1])
        
        # Calculate dz_dvds = ∂(Id/Vd)/∂VDS using the same approach
        for vd_idx in range(len(unique_vd)):
            for vg_idx in range(len(unique_vg)):
                if vd_idx == 0:
                    # Forward difference
                    dz_dvds_grid[vd_idx, vg_idx] = (z_grid[vd_idx+1, vg_idx] - z_grid[vd_idx, vg_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx])
                elif vd_idx == len(unique_vd) - 1:
                    # Backward difference
                    dz_dvds_grid[vd_idx, vg_idx] = (z_grid[vd_idx, vg_idx] - z_grid[vd_idx-1, vg_idx]) / (unique_vd[vd_idx] - unique_vd[vd_idx-1])
                else:
                    # Central difference
                    dz_dvds_grid[vd_idx, vg_idx] = (z_grid[vd_idx+1, vg_idx] - z_grid[vd_idx-1, vg_idx]) / (unique_vd[vd_idx+1] - unique_vd[vd_idx-1])
        
        # Convert grids back to flat arrays matching original data points
        gd_flat = np.zeros_like(id_values)
        gm_flat = np.zeros_like(id_values)
        dz_dvds_flat = np.zeros_like(id_values)
        
        for i, (vg_val, vd_val) in enumerate(zip(vg, vd)):
            vg_idx = np.where(unique_vg == vg_val)[0][0]
            vd_idx = np.where(unique_vd == vd_val)[0][0]
            gd_flat[i] = gd_grid[vd_idx, vg_idx]
            gm_flat[i] = gm_grid[vd_idx, vg_idx]
            dz_dvds_flat[i] = dz_dvds_grid[vd_idx, vg_idx]
        
        print("Derivative calculation complete.")
        return gd_flat, gm_flat, dz_dvds_flat
    
    def inverse_transform_inputs(self, X_scaled):
        """Convert scaled inputs back to original values"""
        return self.x_scaler.inverse_transform(X_scaled)
    
    def get_input_scaler(self):
        """Return the scaler for inputs"""
        return self.x_scaler
    
    def save_to_csv(self, X_train, X_test, y_train, y_test, X_train_orig, X_test_orig, id_train, id_test):
        """Save the preprocessed data to CSV format"""
        # Save all processed data to CSV files
        pd.DataFrame(X_train).to_csv('X_train.csv', index=False, header=False)
        pd.DataFrame(X_test).to_csv('X_test.csv', index=False, header=False)
        pd.DataFrame(y_train).to_csv('y_train.csv', index=False, header=False)
        pd.DataFrame(y_test).to_csv('y_test.csv', index=False, header=False)
        pd.DataFrame(X_train_orig).to_csv('X_train_orig.csv', index=False, header=False)
        pd.DataFrame(X_test_orig).to_csv('X_test_orig.csv', index=False, header=False)
        pd.DataFrame(id_train).to_csv('id_train.csv', index=False, header=False)
        pd.DataFrame(id_test).to_csv('id_test.csv', index=False, header=False)
        print("Data saved in CSV format.")
    
    def save_to_pickle(self, X_train, X_test, y_train, y_test, X_train_orig, X_test_orig, 
                       id_train, id_test, gd_train=None, gd_test=None, gm_train=None, 
                       gm_test=None, dz_dvds_train=None, dz_dvds_test=None):
        """Save the preprocessed data to pickle format"""
        # Create a dictionary of data to save
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_orig': X_train_orig,
            'X_test_orig': X_test_orig,
            'id_train': id_train,
            'id_test': id_test,
            'x_scaler': self.x_scaler  # Added scaler to saved data
        }
        
        # Add derivative data if provided
        if gd_train is not None and gd_test is not None:
            data_dict['gd_train'] = gd_train
            data_dict['gd_test'] = gd_test
        
        if gm_train is not None and gm_test is not None:
            data_dict['gm_train'] = gm_train
            data_dict['gm_test'] = gm_test
        
        if dz_dvds_train is not None and dz_dvds_test is not None:
            data_dict['dz_dvds_train'] = dz_dvds_train
            data_dict['dz_dvds_test'] = dz_dvds_test
        
        # Save data as a pickle file
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        print("Data saved in pickle format.")
        print(f"Data saved to 'preprocessed_data.pkl' and CSV files.")