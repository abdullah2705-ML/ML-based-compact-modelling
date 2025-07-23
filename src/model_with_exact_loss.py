import tensorflow as tf
import numpy as np
import os
from exact_custom_loss import custom_loss_exact

class ExactLossModel:
    def __init__(self, hidden_layers=(8, 8, 8), learning_rate=5e-4, term_selection=1234):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.term_selection = term_selection
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def build_model(self):
        """Build the neural network model"""
        inputs = tf.keras.Input(shape=(2,))
        x = inputs
        
        # Add hidden layers
        for units in self.hidden_layers:
            x = tf.keras.layers.Dense(units, activation='tanh')(x)
        
        # Output layer (no activation - we'll apply transformation later)
        outputs = tf.keras.layers.Dense(1)(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    @tf.function
    def train_step(self, x_batch, y_batch, dz_dvds_batch=None):
        """Perform a single training step with exact loss calculation"""
        # Make sure inputs are float32
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.cast(y_batch, tf.float32)
        if dz_dvds_batch is not None:
            dz_dvds_batch = tf.cast(dz_dvds_batch, tf.float32)
        
        with tf.GradientTape() as tape:
            # Calculate loss using the exact custom loss function with selected terms
            total_loss, terms = custom_loss_exact(
                self.model, x_batch, y_batch, 
                dz_dvds_true=dz_dvds_batch,
                term_selection=self.term_selection, 
                training=True
            )
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, terms
    
    def train(self, data, epochs=500, batch_size=32, fresh_start=False):
        """Train the model with exact custom loss
        
        Args:
            data: Dictionary containing training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            fresh_start: If True, starts training from scratch even if a previous model exists
        """
        # Extract data
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Extract derivatives for exact loss term 4
        dz_dvds_train = data.get('dz_dvds_train', None)
        dz_dvds_test = data.get('dz_dvds_test', None)
        
        # Create datasets with derivatives
        if dz_dvds_train is not None:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                X_train, y_train, dz_dvds_train
            )).shuffle(len(X_train)).batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((
                X_train, y_train
            )).shuffle(len(X_train)).batch(batch_size)
        
        if dz_dvds_test is not None:
            test_dataset = tf.data.Dataset.from_tensor_slices((
                X_test, y_test, dz_dvds_test
            )).batch(batch_size)
        else:
            test_dataset = tf.data.Dataset.from_tensor_slices((
                X_test, y_test
            )).batch(batch_size)
        
        # Training variables
        num_samples = len(X_train)
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 30  # Early stopping patience
        
        # For history tracking
        history = {
            'loss': [],
            'val_loss': [],
            'term1': [],
            'term2': [],
            'term3': [],
            'term4': []
        }
        
        # Get term name based on selection - fixed naming convention
        term_name_map = {
            0: "MSE",
            1: "Term1",
            12: "Term1+2", 
            123: "Term1+2+3",
            1234: "Full"
        }
        term_name = term_name_map.get(self.term_selection, f"Custom({self.term_selection})")
        
        print(f"Training with {term_name} loss: {self.hidden_layers} hidden layers, {self.learning_rate} learning rate")
        
        # Create model directory
        os.makedirs('models', exist_ok=True)
        
        # Define best model path
        best_model_path = f'models/iv_model_exact_loss_{term_name}_best.keras'
        
        # Check if we should load an existing best model to continue from
        if not fresh_start and os.path.exists(best_model_path):
            try:
                self.load_model(best_model_path)
                print(f"Continuing training from existing model: {best_model_path}")
            except Exception as e:
                print(f"Could not load existing model: {e}")
                print("Starting with fresh weights")
        else:
            print(f"Starting with fresh weights for {term_name} loss")
        
        # Main training loop
        # Main training loop
        for epoch in range(epochs):
            # Training loop
            epoch_loss = 0
            epoch_terms = {'term1': 0, 'term2': 0, 'term3': 0, 'term4': 0}
            
            # Handle dataset with or without derivatives
            for step, data_batch in enumerate(train_dataset):
                if len(data_batch) == 3:  # If dataset includes derivatives
                    x_batch, y_batch, dz_dvds_batch = data_batch
                    loss, terms = self.train_step(x_batch, y_batch, dz_dvds_batch)
                else:  # If dataset doesn't include derivatives
                    x_batch, y_batch = data_batch
                    loss, terms = self.train_step(x_batch, y_batch)
                    
                epoch_loss += loss
                
                for term_name_key, term_value in terms.items():
                    epoch_terms[term_name_key] += term_value
                
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}/{steps_per_epoch}, Loss: {loss:.6f}", end='\r')
            
            # Calculate average epoch loss and terms
            avg_train_loss = epoch_loss / steps_per_epoch
            for term_name_key in epoch_terms:
                epoch_terms[term_name_key] /= steps_per_epoch
            
            # Validation loop
            val_loss = 0
            val_terms = {'term1': 0, 'term2': 0, 'term3': 0, 'term4': 0}
            val_steps = 0

            for data_batch in test_dataset:
                # Handle dataset with or without derivatives
                if len(data_batch) == 3:  # If dataset includes derivatives
                    x_batch, y_batch, dz_dvds_batch = data_batch
                    # Calculate validation loss
                    batch_loss, batch_terms = custom_loss_exact(
                        self.model, x_batch, y_batch, 
                        dz_dvds_true=dz_dvds_batch,  # Pass derivatives
                        term_selection=self.term_selection, 
                        training=False
                    )
                else:  # If dataset doesn't include derivatives
                    x_batch, y_batch = data_batch
                    # Calculate validation loss
                    batch_loss, batch_terms = custom_loss_exact(
                        self.model, x_batch, y_batch,
                        term_selection=self.term_selection, 
                        training=False
                    )
                
                val_loss += batch_loss
                
                for term_name_key, term_value in batch_terms.items():
                    val_terms[term_name_key] += term_value
                    
                val_steps += 1
            
            # Calculate average validation loss and terms
            avg_val_loss = val_loss / val_steps
            for term_name_key in val_terms:
                val_terms[term_name_key] /= val_steps
            
            # Record history
            history['loss'].append(float(avg_train_loss))
            history['val_loss'].append(float(avg_val_loss))
            for term_name_key in epoch_terms:
                history[term_name_key].append(float(epoch_terms[term_name_key]))
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
                print(f"  Terms: t1={epoch_terms['term1']:.4f}, t2={epoch_terms['term2']:.4f}, t3={epoch_terms['term3']:.4f}, t4={epoch_terms['term4']:.4f}")
            
            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model using appropriate filename
                self.save_model(best_model_path)
                print(f"Model improved - saving weights to {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model before returning
        if os.path.exists(best_model_path):
            self.load_model(best_model_path)
        else:
            print(f"Warning: Best model file {best_model_path} not found")
            
        return history
    
    def predict(self, x):
        """Make predictions with the model"""
        x = tf.cast(x, tf.float32)
        return self.model(x, training=False)
    
    def save_model(self, filepath):
        """Save the model"""
        # Use the .keras extension for the complete model
        try:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
        
    def load_model(self, filepath):
        """Load the model"""
        if not os.path.exists(filepath):
            print(f"Error: Model file {filepath} does not exist")
            return False
            
        try:
            # Load the complete model
            self.model = tf.keras.models.load_model(filepath, compile=False)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return False