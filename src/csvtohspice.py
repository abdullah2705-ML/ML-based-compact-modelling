import pandas as pd
import os

def csv_to_hspice_table(input_csv, output_sp):
    """Convert CSV data to HSPICE tabular format with corrected syntax"""
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Open output file
    with open(output_sp, 'w') as f:
        # Write HSPICE header
        f.write('* HSPICE script to plot NN model comparison\n')
        f.write('* Generated from CSV data with fixed syntax\n\n')
        f.write('.option post=2 ingold=2 numdgt=10\n\n')
        f.write('.title "Neural Network Model vs Actual MOSFET Comparison"\n\n')
        f.write('* Define temperature\n')
        f.write('.temp 25\n\n')
        
        # Write actual data table
        f.write('* Tabular Data for Actual Id values\n')
        f.write('.data actdata vg vd id\n')
        for _, row in df.iterrows():
            f.write(f"{row['Vg']:.2f} {row['Vd']:.2f} {row['Id_actual']:.12e}\n")
        f.write('.enddata\n\n')
        
        # Write predicted data table
        f.write('* Tabular Data for Predicted Id values\n')
        f.write('.data preddata vg vd id\n')
        for _, row in df.iterrows():
            f.write(f"{row['Vg']:.2f} {row['Vd']:.2f} {row['Id_predicted']:.12e}\n")
        f.write('.enddata\n\n')
        
        # Write dummy circuit components
        f.write('* Create voltage sources for reference and to drive lookup\n')
        f.write('vg g 0 dc=0 sweep vg 0 0.8 0.01\n')
        f.write('vd d 0 dc=0 sweep vd 0 0.8 0.01\n')
        f.write('vs s 0 dc 0\n')
        f.write('vb b 0 dc 0\n\n')
        
        # Use behavioral sources for actual and predicted currents
        f.write('* Behavioral sources to generate currents from table data\n')
        f.write('bact_vd01 act_vd01 0 i=lut2d(actdata, v(g), 0.1)\n')
        f.write('bpred_vd01 pred_vd01 0 i=lut2d(preddata, v(g), 0.1)\n')
        f.write('bact_vd04 act_vd04 0 i=lut2d(actdata, v(g), 0.4)\n')
        f.write('bpred_vd04 pred_vd04 0 i=lut2d(preddata, v(g), 0.4)\n')
        f.write('bact_vd08 act_vd08 0 i=lut2d(actdata, v(g), 0.8)\n')
        f.write('bpred_vd08 pred_vd08 0 i=lut2d(preddata, v(g), 0.8)\n\n')
        
        f.write('bact_vg03 act_vg03 0 i=lut2d(actdata, 0.3, v(d))\n')
        f.write('bpred_vg03 pred_vg03 0 i=lut2d(preddata, 0.3, v(d))\n')
        f.write('bact_vg05 act_vg05 0 i=lut2d(actdata, 0.5, v(d))\n')
        f.write('bpred_vg05 pred_vg05 0 i=lut2d(preddata, 0.5, v(d))\n')
        f.write('bact_vg08 act_vg08 0 i=lut2d(actdata, 0.8, v(d))\n')
        f.write('bpred_vg08 pred_vg08 0 i=lut2d(preddata, 0.8, v(d))\n\n')
        
        # Define dummy MOSFET (reference only)
        f.write('* Define dummy MOSFET for reference (not used in simulation)\n')
        f.write('X1 d g s b nmos_lvt L=20n W=100n\n\n')
        
        # Generate DC analysis statements
        f.write('* DC Analysis sweeps\n')
        f.write('.dc vg 0 0.8 0.01 vd 0.1 0.1 0.1\n')
        f.write('.dc vg 0 0.8 0.01 vd 0.4 0.4 0.1\n')
        f.write('.dc vg 0 0.8 0.01 vd 0.8 0.8 0.1\n')
        f.write('.dc vd 0 0.8 0.01 vg 0.3 0.3 0.1\n')
        f.write('.dc vd 0 0.8 0.01 vg 0.5 0.5 0.1\n')
        f.write('.dc vd 0 0.8 0.01 vg 0.8 0.8 0.1\n\n')
        
        # Generate probe statements for Id vs Vg curves
        f.write('* Plot statements for Id vs Vg at different Vd values\n')
        f.write('.probe dc id_act_vd01=i(bact_vd01)\n')
        f.write('.probe dc id_pred_vd01=i(bpred_vd01)\n')
        f.write('.probe dc id_act_vd04=i(bact_vd04)\n')
        f.write('.probe dc id_pred_vd04=i(bpred_vd04)\n')
        f.write('.probe dc id_act_vd08=i(bact_vd08)\n')
        f.write('.probe dc id_pred_vd08=i(bpred_vd08)\n\n')
        
        # Generate probe statements for Id vs Vd curves
        f.write('* Plot statements for Id vs Vd at different Vg values\n')
        f.write('.probe dc id_act_vg03=i(bact_vg03)\n')
        f.write('.probe dc id_pred_vg03=i(bpred_vg03)\n')
        f.write('.probe dc id_act_vg05=i(bact_vg05)\n')
        f.write('.probe dc id_pred_vg05=i(bpred_vg05)\n')
        f.write('.probe dc id_act_vg08=i(bact_vg08)\n')
        f.write('.probe dc id_pred_vg08=i(bpred_vg08)\n\n')
        
        # Error calculations
        f.write('* Error calculations as percentage\n')
        f.write('.probe dc err_vd01=abs(i(bact_vd01)-i(bpred_vd01))/i(bact_vd01)*100\n')
        f.write('.probe dc err_vd04=abs(i(bact_vd04)-i(bpred_vd04))/i(bact_vd04)*100\n')
        f.write('.probe dc err_vd08=abs(i(bact_vd08)-i(bpred_vd08))/i(bact_vd08)*100\n')
        f.write('.probe dc err_vg03=abs(i(bact_vg03)-i(bpred_vg03))/i(bact_vg03)*100\n')
        f.write('.probe dc err_vg05=abs(i(bact_vg05)-i(bpred_vg05))/i(bact_vg05)*100\n')
        f.write('.probe dc err_vg08=abs(i(bact_vg08)-i(bpred_vg08))/i(bact_vg08)*100\n\n')
        
        # Add measurement statements with correct syntax
        f.write('* Measure statements for key points\n')
        f.write('.meas dc id_act_lin find i(bact_vd01) when vg=0.8\n')
        f.write('.meas dc id_pred_lin find i(bpred_vd01) when vg=0.8\n')
        f.write('.meas dc err_lin_pct param=\'abs(id_act_lin-id_pred_lin)/id_act_lin*100\'\n\n')
        
        f.write('.meas dc id_act_sat find i(bact_vd08) when vg=0.8\n')
        f.write('.meas dc id_pred_sat find i(bpred_vd08) when vg=0.8\n')
        f.write('.meas dc err_sat_pct param=\'abs(id_act_sat-id_pred_sat)/id_act_sat*100\'\n\n')
        
        f.write('.meas dc id_act_sub find i(bact_vd01) when vg=0.2\n')
        f.write('.meas dc id_pred_sub find i(bpred_vd01) when vg=0.2\n')
        f.write('.meas dc err_sub_pct param=\'abs(id_act_sub-id_pred_sub)/id_act_sub*100\'\n\n')
        
        # Add operation point analysis (required by HSPICE)
        f.write('* Analysis - needed for operation\n')
        f.write('.option post=1\n')
        f.write('.op\n\n')
        
        # End the file
        f.write('.end\n')
    
    print(f"HSPICE file created: {output_sp}")

if __name__ == "__main__":
    # Use current directory for input/output
    input_file = "iv_comparison_sorted.csv"
    output_file = "nn_model_comparison_fixed.sp"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
    else:
        csv_to_hspice_table(input_file, output_file)