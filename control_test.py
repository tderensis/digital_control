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

(sys_d_ol, phia, gammaa, L1, L2, K) = control_design.design_tsob(sys_c_ol, T, Ts)
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

# Inverted pendulum 
# (M)       mass of the cart                         (kg)
# (m)       mass of the pendulum                     (kg)
# (b)       coefficient of friction for cart         (N/m/sec)
# (l)       length to pendulum center of mass        (m)
# (I)       mass moment of inertia of the pendulum   (kg.m^2)
# (F)       force applied to the cart                (kg.m/s^2)
# (x)       cart position coordinate                 (m)
# (theta)   pendulum angle from vertical (down)      (rad)

M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

T = 0.01 # sampling time
Ts = 1 # settling time

p = I * (M + m) + M * m * l * l # denominator for the A and B matrices

#"""
A = [   [0,      1,              0,           0],
        [0, -(I+m*l*l)*b/p,  (m*m*g*l*l)/p,   0],
        [0,      0,              0,           1],
        [0, -(m*l*b)/p,       m*g*l*(M+m)/p,  0]]
B = [   [0],
        [(I+m*l*l)/p],
        [0],
        [m*l/p]]
"""
A = [   [0, (I+m*l*l)/p,  (m*m*g*l*l)/p,     0],
        [0,      1,              0,          0],
        [0,      0,              0,          1],
        [0,     m*l/p,       m*g*l*(M+m)/p,  0]]
B = [   [-(I+m*l*l)*b/p],
        [0],
        [0],
        [-(m*l*b)/p]]
#"""
C = [ [1, 0, 0, 0],
      [0, 0, 1, 0]]
D = [[0],
     [0]]

C = np.matrix(C)
D = np.matrix(D)

sys_c_ol = signal.StateSpace(A, B, C, D)
print(sys_c_ol)
(sys_d_ol, L, K) = control_design.design_regob(sys_c_ol, T, Ts)
phi = sys_d_ol.A
gamma = sys_d_ol.B
#print(sys_d_ol)
print("L = ", L)
print("K = ", K)
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regob(sys_d_ol, L, K)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

x0 = [0, 0, 0.1, 0]
(t, u, x, xhat, y) = control_sim.sim_regob(phi, gamma, C, L, K, T, x0, Ts*2)
control_plot.plot_regob(t, u, x, xhat, y)