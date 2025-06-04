import pandas as pd
import numpy as np
import os

def create_simplified_hspice_file(csv_file='iv_comparison_sorted.csv', output_file='iv_final_complete.sp'):
    """
    Create a simplified HSPICE file that works with older HSPICE versions
    using separate DC sweeps for each Vg value
    
    Args:
        csv_file: Input CSV file with Vg, Vd, Id_actual, Id_predicted
        output_file: Output HSPICE file
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully read {csv_file} with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # Check if required columns exist
    required_cols = ['Vg', 'Vd', 'Id_actual', 'Id_predicted']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return False
    
    # Get unique Vg values
    vg_values = sorted(df['Vg'].unique())
    print(f"Found {len(vg_values)} unique Vg values")
    
    # Create the HSPICE header
    hspice_file = "* HSPICE File for IV Comparison - Complete Dataset\n"
    hspice_file += "* Generated from iv_comparison_sorted.csv\n\n"
    
    # Include the standard header
    hspice_file += ".option abstol=1e-6 reltol=1e-6 post ingold numdgt=8\n"
    hspice_file += ".hdl \"/home/2025_spring/abdullah27/perl5/code/bsimcmg.va\"\n"
    hspice_file += ".include \"/home/2025_spring/abdullah27/Research/7nm_FF_160803.pm\"\n\n"
    
    # Add parameters
    hspice_file += "* Parameters\n"
    hspice_file += ".param vdd = 0.8\n\n"
    
    # Set up the original MOSFET circuit
    hspice_file += "******************************************************************\n"
    hspice_file += "* Original MOSFET Circuit\n"
    hspice_file += "******************************************************************\n"
    hspice_file += "* Voltage sources for MOSFET\n"
    hspice_file += "vd_orig d_orig 0 dc = 0\n"
    hspice_file += "vg_orig g_orig 0 dc = 0\n"
    hspice_file += "vs_orig s_orig 0 dc = 0.0\n"
    hspice_file += "vb_orig b_orig 0 dc = 0\n\n"
    hspice_file += "* MOSFET Model (nmos_lvt)\n"
    hspice_file += "X1 d_orig g_orig s_orig b_orig nmos_lvt L=20n W=100n\n\n"
    
    # Set up the neural network model circuit - common components
    hspice_file += "******************************************************************\n"
    hspice_file += "* Common Circuit Components\n"
    hspice_file += "******************************************************************\n"
    hspice_file += "* Vd values for our sweep\n"
    hspice_file += "vd d 0 dc = 0\n\n"
    
    # Sample some Vg values if there are too many to keep the file manageable
    selected_vg_values = vg_values
    if len(vg_values) > 16:
        # Choose a subset of Vg values (e.g., 16 values)
        indices = np.linspace(0, len(vg_values)-1, 16, dtype=int)
        selected_vg_values = [vg_values[i] for i in indices]
        print(f"Selected {len(selected_vg_values)} representative Vg values from the dataset")
    
    # Create DC sweep section
    hspice_file += "******************************************************************\n"
    hspice_file += "* DC Sweeps - One for each selected Vg value\n"
    hspice_file += "******************************************************************\n\n"
    
    # Generate individual DC sweeps for each selected Vg value
    for i, vg in enumerate(selected_vg_values):
        # Format Vg value for display and use in identifiers
        vg_str = f"{vg:.4f}".replace(".", "p")
        
        # Get data for this Vg
        df_vg = df[np.isclose(df['Vg'], vg)].sort_values(by='Vd')
        
        # Skip if we don't have enough data points
        if len(df_vg) < 5:
            print(f"Skipping Vg={vg:.4f} - not enough data points ({len(df_vg)})")
            continue
        
        # Add section header for this Vg sweep
        hspice_file += f"******************************************************************\n"
        hspice_file += f"* DC Sweep - Gate Voltage (Vg={vg:.4f}) and Drain Voltage (Vd)\n"
        hspice_file += f"******************************************************************\n"
        
        # Set parameters and fixed gate voltage for this sweep
        hspice_file += f"* For Vg = {vg:.4f}V\n"
        hspice_file += f".param vg_value = {vg:.4f}\n"
        hspice_file += f"* Fix gate voltage for this sweep\n"
        hspice_file += f"e_vg_fix g_orig 0 dc = {vg:.6f}\n"
        
        # Create PWL current source using the NN predicted current
        hspice_file += f"* Create PWL current source based on data from NN model for this Vg\n"
        hspice_file += f"* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.\n"
        hspice_file += f"i_pred_{vg_str} 0 d pwl("
        
        # Add each Vd,Id_predicted pair to the PWL source
        # HSPICE has line-length limits, so add line breaks
        line_chars = 0
        for j, row in enumerate(df_vg.iterrows()):
            idx, data = row
            vd = data['Vd']
            id_pred = data['Id_predicted']
            
            # Format the point entry
            point_str = f"{vd:.6f} {id_pred:.6e}"
            
            # Check if we need a line break
            if line_chars > 60:
                hspice_file += "\n+ "
                line_chars = 0
            
            # Add the point
            hspice_file += point_str
            line_chars += len(point_str)
            
            # Add space if not the last point
            if j < len(df_vg) - 1:
                hspice_file += " "
                line_chars += 1
        
        # Close the PWL definition
        hspice_file += ")\n"
        
        # Add the load resistor
        hspice_file += f"r_load_{vg_str} d 0 1e6\n"
        
        # Add DC sweep command
        hspice_file += f"* Run DC sweep\n"
        hspice_file += f".dc vd 0 0.8 0.01\n\n"
    
    # Add probe statements
    hspice_file += "******************************************************************\n"
    hspice_file += "* Probe Statements\n"
    hspice_file += "******************************************************************\n"
    hspice_file += "* Probe the gate voltage\n"
    hspice_file += ".probe dc v(g_orig)\n\n"
    hspice_file += "* Probe the drain voltage\n"
    hspice_file += ".probe dc v(d)\n\n"
    hspice_file += "* Probe the actual MOSFET drain current\n"
    hspice_file += ".probe dc i(vd_orig)\n\n"
    
    # Add probes for each PWL current source
    hspice_file += "* Probe the predicted (neural network) currents\n"
    for vg in selected_vg_values:
        vg_str = f"{vg:.4f}".replace(".", "p")
        hspice_file += f".probe dc i(i_pred_{vg_str})\n"
    
    # Add additional options
    hspice_file += "\n* Additional options\n"
    hspice_file += ".option post=2 measout\n\n"
    hspice_file += ".end\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(hspice_file)
    
    # Also create a simplified version with fewer Vg values for easier viewing
    if len(vg_values) > 8:
        # Create a simplified version with only 8 Vg values
        simplified_file = output_file.replace('.sp', '_simplified.sp')
        create_simplified_hspice_file_subset(df, vg_values, simplified_file, 8)
    
    print(f"Created HSPICE file: {output_file}")
    print("Instructions:")
    print("1. Run HSPICE: hspice iv_final_complete.sp")
    print("2. Open WaveView: wv &")
    print("3. In WaveView, for each Vg value, compare:")
    print("   - For actual current: i(vd_orig) vs v(d)")
    print("   - For predicted current: i(i_pred_X) vs v(d) (where X is the Vg value)")
    
    return True

def create_simplified_hspice_file_subset(df, vg_values, output_file, num_vg_values=8):
    """
    Create a simplified HSPICE file with only a subset of Vg values
    for easier viewing and analysis
    """
    # Select a subset of Vg values
    indices = np.linspace(0, len(vg_values)-1, num_vg_values, dtype=int)
    selected_vg_values = [vg_values[i] for i in indices]
    
    # Create the HSPICE header
    hspice_file = "* HSPICE File for IV Comparison - Simplified Version\n"
    hspice_file += f"* Generated with {num_vg_values} selected Vg values for easier viewing\n\n"
    
    # Include the standard header
    hspice_file += ".option abstol=1e-6 reltol=1e-6 post ingold numdgt=8\n"
    hspice_file += ".hdl \"/home/2025_spring/abdullah27/perl5/code/bsimcmg.va\"\n"
    hspice_file += ".include \"/home/2025_spring/abdullah27/Research/7nm_FF_160803.pm\"\n\n"
    
    # Add parameters
    hspice_file += "* Parameters\n"
    hspice_file += ".param vdd = 0.8\n\n"
    
    # Set up the original MOSFET circuit
    hspice_file += "******************************************************************\n"
    hspice_file += "* Original MOSFET Circuit\n"
    hspice_file += "******************************************************************\n"
    hspice_file += "* Voltage sources for MOSFET\n"
    hspice_file += "vd_orig d_orig 0 dc = 0\n"
    hspice_file += "vg_orig g_orig 0 dc = 0\n"
    hspice_file += "vs_orig s_orig 0 dc = 0.0\n"
    hspice_file += "vb_orig b_orig 0 dc = 0\n\n"
    hspice_file += "* MOSFET Model (nmos_lvt)\n"
    hspice_file += "X1 d_orig g_orig s_orig b_orig nmos_lvt L=20n W=100n\n\n"
    
    # Set up the neural network model circuit - common components
    hspice_file += "******************************************************************\n"
    hspice_file += "* Common Circuit Components\n"
    hspice_file += "******************************************************************\n"
    hspice_file += "* Vd values for our sweep\n"
    hspice_file += "vd d 0 dc = 0\n\n"
    
    # Create DC sweep section
    hspice_file += "******************************************************************\n"
    hspice_file += "* DC Sweeps - One for each selected Vg value\n"
    hspice_file += "******************************************************************\n\n"
    
    # Generate individual DC sweeps for each selected Vg value
    for i, vg in enumerate(selected_vg_values):
        # Format Vg value for display and use in identifiers
        vg_str = f"{vg:.4f}".replace(".", "p")
        
        # Get data for this Vg
        df_vg = df[np.isclose(df['Vg'], vg)].sort_values(by='Vd')
        
        # Skip if we don't have enough data points
        if len(df_vg) < 5:
            print(f"Skipping Vg={vg:.4f} - not enough data points ({len(df_vg)})")
            continue
        
        # Sample points to keep the PWL source manageable
        sample_df = df_vg
        if len(df_vg) > 20:
            # Sample every Nth point to get about 20 points
            sample_rate = max(1, len(df_vg) // 20)
            sample_indices = np.arange(0, len(df_vg), sample_rate)
            sample_df = df_vg.iloc[sample_indices].copy()
            
            # Always include the first and last points
            if 0 not in sample_indices:
                sample_df = pd.concat([df_vg.iloc[[0]], sample_df]).reset_index(drop=True)
            if len(df_vg) - 1 not in sample_indices:
                sample_df = pd.concat([sample_df, df_vg.iloc[[-1]]]).reset_index(drop=True)
            
            # Sort again just in case
            sample_df = sample_df.sort_values(by='Vd')
        
        # Add section header for this Vg sweep
        hspice_file += f"******************************************************************\n"
        hspice_file += f"* DC Sweep - Gate Voltage (Vg={vg:.4f}) and Drain Voltage (Vd)\n"
        hspice_file += f"******************************************************************\n"
        
        # Set parameters and fixed gate voltage for this sweep
        hspice_file += f"* For Vg = {vg:.4f}V\n"
        hspice_file += f".param vg_value = {vg:.4f}\n"
        hspice_file += f"* Fix gate voltage for this sweep\n"
        hspice_file += f"e_vg_fix g_orig 0 dc = {vg:.6f}\n"
        
        # Create PWL current source using the NN predicted current
        hspice_file += f"* Create PWL current source based on data from NN model for this Vg\n"
        hspice_file += f"* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.\n"
        hspice_file += f"i_pred_{vg_str} 0 d pwl("
        
        # Add each Vd,Id_predicted pair to the PWL source
        # HSPICE has line-length limits, so add line breaks
        line_chars = 0
        for j, row in enumerate(sample_df.iterrows()):
            idx, data = row
            vd = data['Vd']
            id_pred = data['Id_predicted']
            
            # Format the point entry
            point_str = f"{vd:.6f} {id_pred:.6e}"
            
            # Check if we need a line break
            if line_chars > 60:
                hspice_file += "\n+ "
                line_chars = 0
            
            # Add the point
            hspice_file += point_str
            line_chars += len(point_str)
            
            # Add space if not the last point
            if j < len(sample_df) - 1:
                hspice_file += " "
                line_chars += 1
        
        # Close the PWL definition
        hspice_file += ")\n"
        
        # Add the load resistor
        hspice_file += f"r_load_{vg_str} d 0 1e6\n"
        
        # Add DC sweep command
        hspice_file += f"* Run DC sweep\n"
        hspice_file += f".dc vd 0 0.8 0.01\n\n"
    
    # Add probe statements
    hspice_file += "******************************************************************\n"
    hspice_file += "* Probe Statements\n"
    hspice_file += "******************************************************************\n"
    hspice_file += "* Probe the gate voltage\n"
    hspice_file += ".probe dc v(g_orig)\n\n"
    hspice_file += "* Probe the drain voltage\n"
    hspice_file += ".probe dc v(d)\n\n"
    hspice_file += "* Probe the actual MOSFET drain current\n"
    hspice_file += ".probe dc i(vd_orig)\n\n"
    
    # Add probes for each PWL current source
    hspice_file += "* Probe the predicted (neural network) currents\n"
    for vg in selected_vg_values:
        vg_str = f"{vg:.4f}".replace(".", "p")
        hspice_file += f".probe dc i(i_pred_{vg_str})\n"
    
    # Add additional options
    hspice_file += "\n* Additional options\n"
    hspice_file += ".option post=2 measout\n\n"
    hspice_file += ".end\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(hspice_file)
    
    print(f"Created simplified HSPICE file with {num_vg_values} Vg values: {output_file}")
    print("This simplified version is recommended for easier viewing in WaveView")

if __name__ == "__main__":
    create_simplified_hspice_file()