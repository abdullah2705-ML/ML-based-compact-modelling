* HSPICE File for IV Comparison - Simplified Version
* Generated with 8 selected Vg values for easier viewing

.option abstol=1e-6 reltol=1e-6 post ingold numdgt=8
.hdl "/home/2025_spring/abdullah27/perl5/code/bsimcmg.va"
.include "/home/2025_spring/abdullah27/Research/7nm_FF_160803.pm"

* Parameters
.param vdd = 0.8

******************************************************************
* Original MOSFET Circuit
******************************************************************
* Voltage sources for MOSFET
vd_orig d_orig 0 dc = 0
vg_orig g_orig 0 dc = 0
vs_orig s_orig 0 dc = 0.0
vb_orig b_orig 0 dc = 0

* MOSFET Model (nmos_lvt)
X1 d_orig g_orig s_orig b_orig nmos_lvt L=20n W=100n

******************************************************************
* Common Circuit Components
******************************************************************
* Vd values for our sweep
vd d 0 dc = 0

******************************************************************
* DC Sweeps - One for each selected Vg value
******************************************************************

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.0100) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.0100V
.param vg_value = 0.0100
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.010000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p0100 0 d pwl(0.010000 8.691663e-11 0.050000 2.355278e-10 0.090000 2.815800e-10 
+ 0.130000 3.066067e-10 0.170000 3.253566e-10 0.210000 3.410338e-10 
+ 0.250000 3.551303e-10 0.290000 3.687843e-10 0.330000 3.827743e-10 
+ 0.370000 3.975003e-10 0.410000 4.130719e-10 0.450000 4.294451e-10 
+ 0.490000 4.465209e-10 0.530000 4.642115e-10 0.570000 4.824826e-10 
+ 0.610000 5.013510e-10 0.650000 5.208905e-10 0.690000 5.412171e-10 
+ 0.730000 5.624711e-10 0.770000 5.848012e-10 0.800000 6.023413e-10)
r_load_0p0100 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.1200) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.1200V
.param vg_value = 0.1200
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.120000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p1200 0 d pwl(0.010000 5.517154e-09 0.050000 1.511071e-08 0.090000 1.761428e-08 
+ 0.130000 1.875687e-08 0.170000 1.973998e-08 0.210000 2.072141e-08 
+ 0.250000 2.165849e-08 0.290000 2.251465e-08 0.330000 2.329506e-08 
+ 0.370000 2.403232e-08 0.410000 2.476524e-08 0.450000 2.552479e-08 
+ 0.490000 2.632886e-08 0.530000 2.718276e-08 0.570000 2.808240e-08 
+ 0.610000 2.901672e-08 0.650000 2.997108e-08 0.690000 3.092902e-08 
+ 0.730000 3.187371e-08 0.770000 3.278933e-08 0.800000 3.344809e-08)
r_load_0p1200 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.2300) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.2300V
.param vg_value = 0.2300
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.230000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p2300 0 d pwl(0.010000 2.127103e-07 0.050000 6.734535e-07 0.090000 8.301042e-07 
+ 0.130000 8.862206e-07 0.170000 9.152092e-07 0.210000 9.407795e-07 
+ 0.250000 9.684311e-07 0.290000 9.976372e-07 0.330000 1.026632e-06 
+ 0.370000 1.054220e-06 0.410000 1.080186e-06 0.450000 1.105063e-06 
+ 0.490000 1.129667e-06 0.530000 1.154714e-06 0.570000 1.180595e-06 
+ 0.610000 1.207322e-06 0.650000 1.234579e-06 0.690000 1.261804e-06 
+ 0.730000 1.288289e-06 0.770000 1.313260e-06 0.800000 1.330529e-06)
r_load_0p2300 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.3400) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.3400V
.param vg_value = 0.3400
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.340000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p3400 0 d pwl(0.010000 1.055493e-06 0.050000 4.177544e-06 0.090000 6.021007e-06 
+ 0.130000 7.071860e-06 0.170000 7.658189e-06 0.210000 7.989559e-06 
+ 0.250000 8.192100e-06 0.290000 8.336767e-06 0.330000 8.460128e-06 
+ 0.370000 8.578788e-06 0.410000 8.698845e-06 0.450000 8.821606e-06 
+ 0.490000 8.946672e-06 0.530000 9.073077e-06 0.570000 9.199481e-06 
+ 0.610000 9.323799e-06 0.650000 9.443144e-06 0.690000 9.553748e-06 
+ 0.730000 9.651215e-06 0.770000 9.730949e-06 0.800000 9.776369e-06)
r_load_0p3400 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.4600) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.4600V
.param vg_value = 0.4600
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.460000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p4600 0 d pwl(0.010000 1.714991e-06 0.050000 7.721937e-06 0.090000 1.242053e-05 
+ 0.130000 1.593538e-05 0.170000 1.844300e-05 0.210000 2.015210e-05 
+ 0.250000 2.127714e-05 0.290000 2.201260e-05 0.330000 2.251495e-05 
+ 0.370000 2.289456e-05 0.410000 2.321865e-05 0.450000 2.352018e-05 
+ 0.490000 2.381001e-05 0.530000 2.408738e-05 0.570000 2.434738e-05 
+ 0.610000 2.458521e-05 0.650000 2.479701e-05 0.690000 2.497970e-05 
+ 0.730000 2.513065e-05 0.770000 2.524666e-05 0.800000 2.530856e-05)
r_load_0p4600 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.5700) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.5700V
.param vg_value = 0.5700
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.570000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p5700 0 d pwl(0.010000 2.061635e-06 0.050000 9.716676e-06 0.090000 1.635096e-05 
+ 0.130000 2.190519e-05 0.170000 2.637937e-05 0.210000 2.983825e-05 
+ 0.250000 3.240681e-05 0.290000 3.425430e-05 0.330000 3.556850e-05 
+ 0.370000 3.652787e-05 0.410000 3.727763e-05 0.450000 3.791679e-05 
+ 0.490000 3.849782e-05 0.530000 3.903646e-05 0.570000 3.952660e-05 
+ 0.610000 3.995406e-05 0.650000 4.030648e-05 0.690000 4.057771e-05 
+ 0.730000 4.076767e-05 0.770000 4.088082e-05 0.800000 4.091919e-05)
r_load_0p5700 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.6800) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.6800V
.param vg_value = 0.6800
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.680000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p6800 0 d pwl(0.010000 2.269070e-06 0.050000 1.089627e-05 0.090000 1.872092e-05 
+ 0.130000 2.564933e-05 0.170000 3.161877e-05 0.210000 3.660729e-05 
+ 0.250000 4.064159e-05 0.290000 4.379996e-05 0.330000 4.620750e-05 
+ 0.370000 4.802190e-05 0.410000 4.941216e-05 0.450000 5.053241e-05 
+ 0.490000 5.150069e-05 0.530000 5.238672e-05 0.570000 5.321381e-05 
+ 0.610000 5.397084e-05 0.650000 5.463024e-05 0.690000 5.516468e-05 
+ 0.730000 5.555697e-05 0.770000 5.580337e-05 0.800000 5.589706e-05)
r_load_0p6800 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* DC Sweep - Gate Voltage (Vg=0.8000) and Drain Voltage (Vd)
******************************************************************
* For Vg = 0.8000V
.param vg_value = 0.8000
* Fix gate voltage for this sweep
e_vg_fix g_orig 0 dc = 0.800000
* Create PWL current source based on data from NN model for this Vg
* The format is: Vd1 Id1 Vd2 Id2 Vd3 Id3 etc.
i_pred_0p8000 0 d pwl(0.010000 2.375805e-06 0.050000 1.151707e-05 0.090000 2.001362e-05 
+ 0.130000 2.778564e-05 0.170000 3.476649e-05 0.210000 4.090800e-05 
+ 0.250000 4.618688e-05 0.290000 5.061038e-05 0.330000 5.422210e-05 
+ 0.370000 5.710366e-05 0.410000 5.937148e-05 0.450000 6.116562e-05 
+ 0.490000 6.263168e-05 0.530000 6.389826e-05 0.570000 6.505780e-05 
+ 0.610000 6.615490e-05 0.650000 6.718834e-05 0.690000 6.812488e-05 
+ 0.730000 6.892017e-05 0.770000 6.953610e-05 0.800000 6.986799e-05)
r_load_0p8000 d 0 1e6
* Run DC sweep
.dc vd 0 0.8 0.01

******************************************************************
* Probe Statements
******************************************************************
* Probe the gate voltage
.probe dc v(g_orig)

* Probe the drain voltage
.probe dc v(d)

* Probe the actual MOSFET drain current
.probe dc i(vd_orig)

* Probe the predicted (neural network) currents
.probe dc i(i_pred_0p0100)
.probe dc i(i_pred_0p1200)
.probe dc i(i_pred_0p2300)
.probe dc i(i_pred_0p3400)
.probe dc i(i_pred_0p4600)
.probe dc i(i_pred_0p5700)
.probe dc i(i_pred_0p6800)
.probe dc i(i_pred_0p8000)

* Additional options
.option post=2 measout

.end
