import pandas as pd
import numpy as np
import os

def create_lookup_table(csv_file='iv_comparison_sorted.csv', output_file='nn_current_table.inc'):
    """
    Create a lookuptable for HSPICE that uses piecewise functions
    instead of PWL2D (which isn't supported in some HSPICE versions)
    
    Args:
        csv_file: Input CSV file with Vg, Vd, Id_actual, Id_predicted
        output_file: Output include file for HSPICE
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create the HSPICE header
    hspice_file = "* Neural Network Current Table\n"
    hspice_file += "* This uses B-source lookup tables to implement the neural network model\n\n"
    
    # Add the B-source definition
    hspice_file += "* B-source for the neural network current\n"
    hspice_file += "* Uses table_model() from our CSV data as a lookup table\n"
    hspice_file += "Bnn d_nn s_nn i='table_model(v(g_nn),v(d_nn))'\n\n"
    
    # Add the dummy zero function to handle zero Vd
    hspice_file += "* Make sure we have dummy source for 0 Vd to avoid divide by zero\n"
    hspice_file += ".func dummy_zero(vd) = (vd < 0.001) ? 0 : vd\n\n"
    
    # Get unique Vg values, and round them to 1 decimal place for easier indexing
    # This is important because HSPICE may have numerical precision issues
    df['Vg_rounded'] = df['Vg'].apply(lambda x: round(x * 10) / 10)
    vg_values = sorted(df['Vg_rounded'].unique())
    
    # Select a subset of Vg values to keep the code manageable
    # We'll use approximately 9 Vg values from 0.0 to 0.8
    target_vg_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Find the closest Vg values in our dataset to the target values
    selected_vg = []
    for target in target_vg_values:
        closest = min(vg_values, key=lambda x: abs(x - target))
        selected_vg.append(closest)
    
    # Create the main table_model function
    hspice_file += "* Table model function - we implement as a set of 1D tables, one for each Vg value\n"
    hspice_file += "* This is more compatible with older HSPICE versions than PWL2D\n"
    hspice_file += ".func table_model(vg,vd) = \\\n"
    
    # Add the conditional logic to select the right Vg table
    for i, vg in enumerate(selected_vg[:-1]):  # All but the last
        next_vg = selected_vg[i + 1]
        mid_vg = (vg + next_vg) / 2
        hspice_file += f"+ (vg < {mid_vg:.2f}) ? vg_{int(vg*100):d}(dummy_zero(vd)) : \\\n"
    
    # Add the last Vg value
    last_vg = selected_vg[-1]
    hspice_file += f"+ vg_{int(last_vg*100):d}(dummy_zero(vd))\n\n"
    
    # Now create a table for each selected Vg value
    for vg in selected_vg:
        # Get data for this Vg value
        vg_data = df[np.isclose(df['Vg_rounded'], vg)].sort_values(by='Vd')
        
        # Add header for this Vg table
        hspice_file += f"* Table for Vg={vg:.1f}\n"
        hspice_file += f".func vg_{int(vg*100):d}(vd) = \\\n"
        
        # Handle the case of Vd=0 explicitly
        hspice_file += "+ (vd < 0.01) ? 0 : \\\n"
        
        # Add lookup entries for each Vd value
        vd_values = sorted(vg_data['Vd'].unique())
        
        for i, vd in enumerate(vd_values[:-1]):  # All but the last
            if vd < 0.01:  # Skip Vd=0 since we handled it above
                continue
                
            next_vd = vd_values[i + 1]
            
            # Get the Id_predicted for this Vd
            id_val = vg_data[vg_data['Vd'] == vd]['Id_predicted'].values[0]
            
            # Add this entry to the table
            hspice_file += f"+ (vd < {next_vd:.2f}) ? {id_val:.8e} : \\\n"
        
        # Add the last Vd value
        last_vd = vd_values[-1]
        last_id = vg_data[vg_data['Vd'] == last_vd]['Id_predicted'].values[0]
        hspice_file += f"+ {last_id:.8e}\n\n"
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write(hspice_file)
    
    print(f"Created lookup table file for HSPICE in {output_file}")
    print(f"Table covers {len(selected_vg)} Vg values from {selected_vg[0]:.1f} to {selected_vg[-1]:.1f}")
    print("You can now run HSPICE with the iv_comparison_fixed.sp file")
    
    # Create a README with instructions
    with open('how_to_run_hspice_fixed.txt', 'w') as f:
        f.write("# Fixed Instructions for Running and Viewing Results\n\n")
        
        f.write("## 1. Run HSPICE with the fixed file:\n")
        f.write("```\n")
        f.write("hspice iv_comparison_fixed.sp\n")
        f.write("```\n\n")
        
        f.write("## 2. Open WaveView:\n")
        f.write("```\n")
        f.write("wv &\n")
        f.write("```\n\n")
        
        f.write("## 3. Open the .tr0 file in WaveView\n\n")
        
        f.write("## 4. Plot the following Id-Vg curves:\n")
        f.write("- For actual current: `I(vd_mos)` vs `V(g_mos)` for various `V(d_mos)` values\n")
        f.write("- For predicted current: `I(vid_nn)` vs `V(g_nn)` for the same `V(d_nn)` values\n\n")
        
        f.write("## 5. Plot the following Id-Vd curves:\n")
        f.write("- For actual current: `I(vd_mos)` vs `V(d_mos)` for various `V(g_mos)` values\n")
        f.write("- For predicted current: `I(vid_nn)` vs `V(d_nn)` for the same `V(g_nn)` values\n\n")
        
        f.write("## 6. Tips for better visualization:\n")
        f.write("- Use log scale for current (right-click on Y-axis → Scale → Log)\n")
        f.write("- Use different colors for actual and predicted curves\n")
        f.write("- Add grid lines (View → Grid)\n")
        f.write("- Add curve labels (right-click on curve → Add Curve Label)\n")
    
    print("Created instructions in 'how_to_run_hspice_fixed.txt'")
    
    return True

if __name__ == "__main__":
    create_lookup_table()