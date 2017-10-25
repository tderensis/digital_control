"""
A simple test script and general scratch area.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.linalg as LA
import control_plot, control_sim, control_design, control_optimize, control_eval, control_poles

A = [   [0,  1,  0],
        [0, -1,  1],
        [0,  0, -4]]
B = [   [0],
        [0],
        [1]]
C = [1, 0, 0]
D = [ 0 ]
Ts = 2
T = 0.1
Tso = Ts/4

sys_c_ol = signal.StateSpace(A, B, C, D)
C = sys_c_ol.C
D = sys_c_ol.D
initial = [0,1,0]

spoles = control_poles.bessel_spoles(3, Ts)
(sys_d_ol, L, K) = control_design.design_regob(sys_c_ol, T, Ts, spoles=spoles)
phi = sys_d_ol.A
gamma = sys_d_ol.B
print("L = ", L)
print("K = ", K)

print("Full state feedback regulator margins")
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regsf(sys_d_ol, L)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

print("Full order observer regulator margins")
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regob(sys_d_ol, L, K)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

(sys_d_ol, phia, gammaa, L1, L2, K) = control_design.design_tsob(sys_c_ol, T, Ts, Tso)
print("phia = ", phia)
print("gammaa = ", gammaa)
print("L1 = ", L1)
print("L2 = ", L2)
print("Full state feedback tracking system margins")
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_tssf(sys_d_ol, phia, gammaa, L1, L2)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

print("Full order observer tracking system margins")
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_tsob(sys_d_ol, phia, gammaa, L1, L2, K)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

"""
(t, u, x) = control_sim.sim_regsf(phi, gamma, L, T, initial, Ts*2)
print("settling time = ", control_eval.settling_time(t, x))
control_plot.plot_regsf(t, u, x, "bessel", False)

(t, u, x, xhat, y) = control_sim.sim_regob(phi, gamma, C, L, K, T, initial, Ts*2)
control_plot.plot_regob(t, u, x, xhat, y, "observer", True)
"""

"""
# optimized results
L = control_optimize.optimize_regsf(phi, gamma, T, Ts)
#L = np.array([ 18.31630424,  12.72504718,   1.97528811])
print("L = ", L)
control_eval.print_stability_margins(phi, gamma, L)
(t, u, x) = control_sim.sim_regsf(phi, gamma, L, T, initial, Ts*2)
print("settling time = ", control_eval.settling_time(t, x))
control_plot.plot_regsf(t, u, x, "optimized", True)
"""

