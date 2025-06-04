import pandas as pd
import numpy as np
import os

def create_fixed_hspice_plot_file(csv_file='iv_comparison.csv', output_file='iv_data_for_wv_fixed.sp'):
    """
    Create an HSPICE file for plotting that avoids the voltage source loop error
    by using independent sources and behavioral sources
    
    Args:
        csv_file: Input CSV file with Vg, Vd, Id_actual, Id_predicted
        output_file: Output HSPICE file for WaveView
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create the HSPICE header
    hspice_file = "* HSPICE Data File for Plotting Id-Vg and Id-Vd Curves in WaveView\n"
    hspice_file += ".option post=2 ingold=2 numdgt=10\n\n"
    
    # Define independent time source for simulation
    hspice_file += "* Independent time source for simulation\n"
    hspice_file += "VTIME time 0 PWL(0 0 1 1)\n\n"
    
    # Create separate data tables for each Vd value (for Id-Vg curves)
    vd_values = sorted(df['Vd'].unique())
    
    # Create data tables for Id-Vg curves
    for vd_val in vd_values:
        # Filter data for this Vd
        df_vd = df[df['Vd'] == vd_val].sort_values(by='Vg')
        
        if len(df_vd) > 0:
            # Create tables for actual and predicted data
            hspice_file += f"* Data table for Id-Vg at Vd = {vd_val:.2f}V\n"
            
            # Table for VG values
            hspice_file += f".param vg_data_vd{vd_val:.2f}=[ \\\n"
            for i, row in enumerate(df_vd.iterrows()):
                _, data = row
                hspice_file += f"+ {data['Vg']:.6f}"
                if i < len(df_vd) - 1:
                    hspice_file += ", "
                # Add newlines every 10 values for readability
                if (i + 1) % 10 == 0 and i < len(df_vd) - 1:
                    hspice_file += " \\\n"
            hspice_file += " ]\n\n"
            
            # Table for actual Id values
            hspice_file += f".param id_actual_data_vd{vd_val:.2f}=[ \\\n"
            for i, row in enumerate(df_vd.iterrows()):
                _, data = row
                hspice_file += f"+ {data['Id_actual']:.12e}"
                if i < len(df_vd) - 1:
                    hspice_file += ", "
                # Add newlines every 5 values for readability
                if (i + 1) % 5 == 0 and i < len(df_vd) - 1:
                    hspice_file += " \\\n"
            hspice_file += " ]\n\n"
            
            # Table for predicted Id values
            hspice_file += f".param id_pred_data_vd{vd_val:.2f}=[ \\\n"
            for i, row in enumerate(df_vd.iterrows()):
                _, data = row
                hspice_file += f"+ {data['Id_predicted']:.12e}"
                if i < len(df_vd) - 1:
                    hspice_file += ", "
                # Add newlines every 5 values for readability
                if (i + 1) % 5 == 0 and i < len(df_vd) - 1:
                    hspice_file += " \\\n"
            hspice_file += " ]\n\n"
    
    # Create data tables for Id-Vd curves
    vg_values = sorted(df['Vg'].unique())
    
    for vg_val in vg_values:
        # Filter data for this Vg
        df_vg = df[df['Vg'] == vg_val].sort_values(by='Vd')
        
        if len(df_vg) > 0:
            # Create tables for actual and predicted data
            hspice_file += f"* Data table for Id-Vd at Vg = {vg_val:.2f}V\n"
            
            # Table for VD values
            hspice_file += f".param vd_data_vg{vg_val:.2f}=[ \\\n"
            for i, row in enumerate(df_vg.iterrows()):
                _, data = row
                hspice_file += f"+ {data['Vd']:.6f}"
                if i < len(df_vg) - 1:
                    hspice_file += ", "
                # Add newlines every 10 values for readability
                if (i + 1) % 10 == 0 and i < len(df_vg) - 1:
                    hspice_file += " \\\n"
            hspice_file += " ]\n\n"
            
            # Table for actual Id values
            hspice_file += f".param id_actual_data_vg{vg_val:.2f}=[ \\\n"
            for i, row in enumerate(df_vg.iterrows()):
                _, data = row
                hspice_file += f"+ {data['Id_actual']:.12e}"
                if i < len(df_vg) - 1:
                    hspice_file += ", "
                # Add newlines every 5 values for readability
                if (i + 1) % 5 == 0 and i < len(df_vg) - 1:
                    hspice_file += " \\\n"
            hspice_file += " ]\n\n"
            
            # Table for predicted Id values
            hspice_file += f".param id_pred_data_vg{vg_val:.2f}=[ \\\n"
            for i, row in enumerate(df_vg.iterrows()):
                _, data = row
                hspice_file += f"+ {data['Id_predicted']:.12e}"
                if i < len(df_vg) - 1:
                    hspice_file += ", "
                # Add newlines every 5 values for readability
                if (i + 1) % 5 == 0 and i < len(df_vg) - 1:
                    hspice_file += " \\\n"
            hspice_file += " ]\n\n"
    
    # Create behavioral voltage sources for plotting Id-Vg data
    hspice_file += "* Behavioral voltage sources for Id-Vg plots\n"
    
    # Create sources for a few selected Vd values (to avoid too many signals)
    selected_vd_values = vd_values
    if len(vd_values) > 5:
        # Select approximately 5 evenly spaced values
        indices = np.linspace(0, len(vd_values)-1, 5, dtype=int)
        selected_vd_values = [vd_values[i] for i in indices]
    
    for vd_val in selected_vd_values:
        # Count number of points for this Vd
        df_vd = df[df['Vd'] == vd_val].sort_values(by='Vg')
        n_points = len(df_vd)
        
        if n_points > 0:
            # Fixed the string formatting issues - properly escape curly braces in f-strings
            # VG for this Vd
            vd_str = f"{vd_val:.2f}"
            hspice_file += f"EVG_VD{vd_str} vg_vd{vd_str} 0 VOL='vg_data_vd{vd_str}[min(floor(time*{n_points}),{n_points-1})]'\n"
            
            # Actual and predicted Id
            hspice_file += f"EID_ACT_VD{vd_str} id_act_vd{vd_str} 0 VOL='id_actual_data_vd{vd_str}[min(floor(time*{n_points}),{n_points-1})]'\n"
            hspice_file += f"EID_PRED_VD{vd_str} id_pred_vd{vd_str} 0 VOL='id_pred_data_vd{vd_str}[min(floor(time*{n_points}),{n_points-1})]'\n\n"
    
    # Create behavioral voltage sources for plotting Id-Vd data
    hspice_file += "* Behavioral voltage sources for Id-Vd plots\n"
    
    # Create sources for a few selected Vg values (to avoid too many signals)
    selected_vg_values = vg_values
    if len(vg_values) > 5:
        # Select approximately 5 evenly spaced values
        indices = np.linspace(0, len(vg_values)-1, 5, dtype=int)
        selected_vg_values = [vg_values[i] for i in indices]
    
    for vg_val in selected_vg_values:
        # Count number of points for this Vg
        df_vg = df[df['Vg'] == vg_val].sort_values(by='Vd')
        n_points = len(df_vg)
        
        if n_points > 0:
            # Fixed the string formatting issues - properly escape curly braces in f-strings
            # VD for this Vg
            vg_str = f"{vg_val:.2f}"
            hspice_file += f"EVD_VG{vg_str} vd_vg{vg_str} 0 VOL='vd_data_vg{vg_str}[min(floor(time*{n_points}),{n_points-1})]'\n"
            
            # Actual and predicted Id
            hspice_file += f"EID_ACT_VG{vg_str} id_act_vg{vg_str} 0 VOL='id_actual_data_vg{vg_str}[min(floor(time*{n_points}),{n_points-1})]'\n"
            hspice_file += f"EID_PRED_VG{vg_str} id_pred_vg{vg_str} 0 VOL='id_pred_data_vg{vg_str}[min(floor(time*{n_points}),{n_points-1})]'\n\n"
    
    # Add transient analysis
    hspice_file += "* Run a transient analysis to plot the data\n"
    hspice_file += ".tran 0.01 1.0\n\n"
    
    # Add plotting hints
    hspice_file += "* Plotting hints for WaveView:\n"
    hspice_file += "* For Id-Vg plots:\n"
    for vd_val in selected_vd_values:
        vd_str = f"{vd_val:.2f}"
        hspice_file += f"*   For Vd={vd_str}V: Plot V(id_act_vd{vd_str}) vs V(vg_vd{vd_str}) and V(id_pred_vd{vd_str}) vs V(vg_vd{vd_str})\n"
    
    hspice_file += "* For Id-Vd plots:\n"
    for vg_val in selected_vg_values:
        vg_str = f"{vg_val:.2f}"
        hspice_file += f"*   For Vg={vg_str}V: Plot V(id_act_vg{vg_str}) vs V(vd_vg{vg_str}) and V(id_pred_vg{vg_str}) vs V(vd_vg{vg_str})\n"
    
    # End the file
    hspice_file += "\n.end\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(hspice_file)
    
    # Create a simple README file with plotting instructions
    with open('waveview_plotting_instructions_fixed.txt', 'w') as f:
        f.write("# Instructions for Plotting in WaveView\n\n")
        
        f.write("## Running the HSPICE file\n")
        f.write("Run the HSPICE file with:\n")
        f.write(f"hspice {output_file}\n\n")
        
        f.write("## Opening in WaveView\n")
        f.write("Open WaveView and load the .tr0 file:\n")
        f.write("wv &\n\n")
        
        f.write("## Plotting Id-Vg Curves\n")
        f.write("For each Vd value:\n")
        for vd_val in selected_vd_values:
            vd_str = f"{vd_val:.2f}"
            f.write(f"- For Vd={vd_str}V:\n")
            f.write(f"  * Plot V(id_act_vd{vd_str}) vs V(vg_vd{vd_str}) for actual current\n")
            f.write(f"  * Plot V(id_pred_vd{vd_str}) vs V(vg_vd{vd_str}) for predicted current\n\n")
        
        f.write("## Plotting Id-Vd Curves\n")
        f.write("For each Vg value:\n")
        for vg_val in selected_vg_values:
            vg_str = f"{vg_val:.2f}"
            f.write(f"- For Vg={vg_str}V:\n")
            f.write(f"  * Plot V(id_act_vg{vg_str}) vs V(vd_vg{vg_str}) for actual current\n")
            f.write(f"  * Plot V(id_pred_vg{vg_str}) vs V(vd_vg{vg_str}) for predicted current\n\n")
        
        f.write("## Tips for Better Visualization\n")
        f.write("1. Use logarithmic scale for current (Y-axis) by right-clicking on Y-axis and selecting 'Log'\n")
        f.write("2. Add grid lines through View → Grid\n")
        f.write("3. Use different colors for actual and predicted curves\n")
        f.write("4. Use the 'Edit → Add curve label' option to add labels to the curves\n")
    
    print(f"Fixed HSPICE file created and saved to {output_file}")
    print(f"Detailed plotting instructions saved to 'waveview_plotting_instructions_fixed.txt'")
    print("\nThis fixed version uses behavioral voltage sources instead of directly connected PWL sources")
    print("to avoid the 'Inductor/voltage source loop' error.")
    
    return True

if __name__ == "__main__":
    create_fixed_hspice_plot_file()