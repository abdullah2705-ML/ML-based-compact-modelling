import tensorflow as tf

def custom_loss_exact(model, x_batch, y_true, dz_dvds_true=None, term_selection=1234, training=True):
    """
    Compute exact custom loss with selected derivative terms as specified in the paper,
    using TensorFlow's automatic differentiation
    
    Args:
        model: The neural network model
        x_batch: Input features (VGS, VDS)
        y_true: True output values (ln(Id/Vd))
        dz_dvds_true: True values of dz/dVDS calculated from data
        term_selection: Which terms to include: 0=MSE, 1=Term1, 12=Terms1+2, 123=Terms1+2+3, 1234=All terms
        training: Whether in training mode
    
    Returns:
        total_loss: The computed loss value
        terms: Dictionary containing individual loss terms
    """
    # Constants for the loss function matching paper values
    a = 1.0    # Weight for term1 
    b = 2.0    # Weight for term2 (gd error)
    c = 700.0  # Weight for term3 (gm error)
    d = 100.0   # Weight for term4 (dz/dVDS error)
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    # Check if we should use MSE loss instead of custom loss
    if term_selection == 0:
        # Simple MSE loss
        y_pred = model(x_batch, training=training)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return mse_loss, {'term1': mse_loss, 'term2': 0.0, 'term3': 0.0, 'term4': 0.0}
    
    # Convert all inputs to tensors and ensure they are float32
    x_batch = tf.cast(x_batch, tf.float32)
    if len(y_true.shape) == 1:
        y_true = tf.expand_dims(y_true, axis=1)
    y_true = tf.cast(y_true, tf.float32)
    
    # Extract VGS and VDS from input batch
    vgs = tf.expand_dims(x_batch[:, 0], axis=1)
    vds = tf.expand_dims(x_batch[:, 1], axis=1)
    
    # Initialize term values
    term1 = 0.0
    term2 = 0.0
    term3 = 0.0
    term4 = 0.0
    
    # Watch vgs and vds for gradient computation
    with tf.GradientTape(persistent=True) as tape_outer:
        tape_outer.watch(vgs)
        tape_outer.watch(vds)
        
        # Combine watched tensors back into input for the model
        x_watched = tf.concat([vgs, vds], axis=1)
        
        with tf.GradientTape(persistent=True) as tape_inner:
            tape_inner.watch(vgs)
            tape_inner.watch(vds)
            
            # Forward pass through the model
            y_pred = model(x_watched, training=training)
            
            # Calculate Id = Vd * e^y for both true and predicted values (Equation 1)
            z_true = tf.exp(y_true)
            z_pred = tf.exp(y_pred)
            
            id_true = vds * z_true
            id_pred = vds * z_pred
        
        # Term 1: ratio error in y (always include if term_selection >= 1)
        if term_selection >= 1:
            ratio_y = (y_true - y_pred) / (tf.abs(y_true) + epsilon)
            term1 = tf.sqrt(tf.reduce_mean(tf.square(ratio_y)))
        
        # ----------- COMPUTE EXACT DERIVATIVES USING AUTODIFF -----------
        
        # For terms 2, 3, and 4, compute them only if selected
        
        # Term 2: ratio error in gd (include if term_selection >= 12)
        if term_selection >= 12:
            # Compute gd = ∂Id/∂VDS (output conductance)
            gd_true_raw = tape_inner.gradient(id_true, vds)
            gd_pred_raw = tape_inner.gradient(id_pred, vds)
            
            # Handle None values (should not happen with proper setup)
            gd_true = tf.zeros_like(y_true) if gd_true_raw is None else gd_true_raw
            gd_pred = tf.zeros_like(y_pred) if gd_pred_raw is None else gd_pred_raw
            
            ratio_gd = (gd_true - gd_pred) / (tf.abs(gd_true) + epsilon)
            term2 = tf.sqrt(tf.reduce_mean(tf.square(ratio_gd)))
        
        # Term 3: absolute error in gm (include if term_selection >= 123)
        if term_selection >= 123:
            # Compute gm = ∂Id/∂VGS (transconductance)
            gm_true_raw = tape_inner.gradient(id_true, vgs)
            gm_pred_raw = tape_inner.gradient(id_pred, vgs)
            
            # Handle None values
            gm_true = tf.zeros_like(y_true) if gm_true_raw is None else gm_true_raw
            gm_pred = tf.zeros_like(y_pred) if gm_pred_raw is None else gm_pred_raw
            
            term3 = tf.sqrt(tf.reduce_mean(tf.square(gm_true - gm_pred)))
    
    # Term 4: absolute error in dz/dVDS (include if term_selection >= 1234)
    if term_selection >= 1234:
        # Compute dz/dVDS for predicted values
        with tf.GradientTape() as tape:
            tape.watch(vds)
            z_pred_for_grad = tf.exp(model(tf.concat([vgs, vds], axis=1), training=training))
        
        dz_pred_dvds = tape.gradient(z_pred_for_grad, vds)
        
        # Use the provided true derivative values instead of zeros
        if dz_dvds_true is not None:
            # Make sure it's a tensor with the right shape
            dz_dvds_true = tf.cast(dz_dvds_true, tf.float32)
            if len(dz_dvds_true.shape) == 1:
                dz_dvds_true = tf.expand_dims(dz_dvds_true, axis=1)
        else:
            # Fallback to zeros if not provided
            dz_dvds_true = tf.zeros_like(y_true)
        
        term4 = tf.sqrt(tf.reduce_mean(tf.square(dz_dvds_true - dz_pred_dvds)))
    
    # Combine terms with weights based on which terms are selected
    # Use tf.clip_by_value to prevent loss explosion
    term1_safe = tf.clip_by_value(term1, 0.0, 10.0)
    term2_safe = tf.clip_by_value(term2, 0.0, 10.0)
    term3_safe = tf.clip_by_value(term3, 0.0, 10.0)
    term4_safe = tf.clip_by_value(term4, 0.0, 10.0)
    
    # Build the total loss based on selected terms
    total_loss = a * term1_safe
    if term_selection >= 12:
        total_loss += b * term2_safe
    if term_selection >= 123:
        total_loss += c * term3_safe
    if term_selection >= 1234:
        total_loss += d * term4_safe
    
    terms = {
        'term1': term1,
        'term2': term2, 
        'term3': term3,
        'term4': term4
    }
    
    return total_loss, terms