"""
Design of a state space controller for an inverted pendulum driven by stepper motor.
"""
import control_plot, control_sim, control_design, control_optimize, control_eval, control_poles
from scipy import signal
import numpy as np
import math


# System Clasification Results

# motor position low pass filter (bessel with 1 sec settling time)
b_1 = 21.9
b_0 = 8.106
b_g = 21.9

g = 9.81
w0 = 4.008 # natural frequency
d = 0.0718 # damping
a_1 = w0**2
a_2 = a_1/g


# State Space Equations
"""
x = | x         | - motor position (m)
    | vel       | - motor velocity (m/s)
    | theta     | - pendulum position (rad)
    | theta_dot | - pendulum velocity (rad/s)

u = | x_d |       - desired motor position (m)
"""
A = np.matrix([
    [ 0,        1,       0,    0],
    [-b_1,     -b_0,     0,    0],
    [ 0,        0,       0,    1],
    [-b_1*a_2, -b_0*a_2, a_1, -d]
])
B = np.matrix([
    [0],
    [b_g],
    [0],
    [b_g*a_2]
])
C = np.matrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
D = np.matrix([
    [0],
    [0]
])

sys_c_ol = signal.StateSpace(A, B, C, D)

print(sys_c_ol)

T = 0.05 # sampling time
Ts = 1.2 # settling time
Tso = Ts/6

print("Using T =", T, "Ts =", Ts, "Tso = ", Tso)
spoles = [
(-4.053+2.34j), (-4.053-2.34j), (-4.044060776465936+0j), (-3.9722607764659337+0j)
]

(sys_d_ol, L, K) = control_design.design_regob(sys_c_ol, T, Ts, Tso, spoles)
phi = sys_d_ol.A
gamma = sys_d_ol.B
print("phi =\n", phi)
print("gamma =\n", gamma)
print("L =\n", L)
print("K =\n", K)

(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regsf(sys_d_ol, L)
print("Stability assuming all states are measured")
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regob(sys_d_ol, L, K)
print("Stability using a full order observer")
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

x0 = np.zeros((1, 4))
x0[0,1] = 20/math.pi
(t, u, x) = control_sim.sim_regsf(phi, gamma, L, T, x0, Ts*2)
print("reg settling time = ", control_eval.settling_time(t, x))
control_plot.plot_regsf(t, u, x)

(t, u, x, xhat, y) = control_sim.sim_regob(phi, gamma, C, L, K, T, x0, Ts*2)
print("fob settling time = ", control_eval.settling_time(t, y))
control_plot.plot_regob(t, u, x, xhat, y)

# Add a pole for the tracking system
spoles = spoles + control_poles.bessel_spoles(1, Ts)

# Only position is tracked
Ca = np.matrix([ 1, 0, 0, 0 ])
(sys_d_ol, phia, gammaa, L1, L2, K) = control_design.design_tsob(sys_c_ol, Ca, T, Ts, Tso, spoles)

print("phia = ", phia)
print("gammaa = ", gammaa)
print("L1 = ", L1)
print("L2 = ", L2)
print("K =\n", K)

(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_tssf(sys_d_ol, phia, gammaa, Ca, L1, L2)
print("Stability using a tracking system with full state feedback.")
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_tsob(sys_d_ol, phia, gammaa, Ca, L1, L2, K)
print("Stability using a tracking system with full order observer")
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)
