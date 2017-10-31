"""
Design of a state space controller for an inverted pendulum.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.linalg as LA
import control_plot, control_sim, control_design, control_optimize, control_eval, control_poles

# Inverted pendulum 
# (M)       mass of the cart                         (kg)
# (m)       mass of the pendulum                     (kg)
# (b)       coefficient of friction for cart         (N/m/sec)
# (l)       length to pendulum center of mass        (m)
# (I)       mass moment of inertia of the pendulum   (kg.m^2)
# (F)       force applied to the cart                (kg.m/s^2)
# (x)       cart position coordinate                 (m)
# (theta)   pendulum angle from vertical (down)      (rad)

M = 0.5
m = 0.2
b = 1
g = 9.8
l = 0.3
I = 1/3 * l**2 * m # assumes a rod

T = 0.02 # sampling time
Ts = 2 # settling time
Tso = Ts/6
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
spoles=None
(sys_d_ol, L, K) = control_design.design_regob(sys_c_ol, T, Ts, Tso, spoles=spoles)
phi = sys_d_ol.A
gamma = sys_d_ol.B
print("L = ", L)
print("K = ", K)

print("Stability assuming all states are measured")
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regsf(sys_d_ol, L)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

print("Stability using a full order observer")
(phi_ltf, gamma_ltf, c_ltf) = control_eval.ltf_regob(sys_d_ol, L, K)
control_eval.print_stability_margins(phi_ltf, gamma_ltf, c_ltf)

x0 = [0.1, 0, 0.1, 0]
(t, u, x, xhat, y) = control_sim.sim_regob(phi, gamma, C, L, K, T, x0, Ts*2)
print("settling time = ", control_eval.settling_time(t, y))
#control_plot.plot_regob(t, u, x, xhat, y)


import serial, io, math
from time import sleep

def readline(a_serial, eol=b'\n'):
    leneol = len(eol)
    line = bytearray()
    while True:
        c = a_serial.read(1)
        if c:
            line += c
            if line[-leneol:] == eol:
                break
        else:
            break
    return bytes(line)
    
microsteps=1.0
input('Press Enter to start')
# start serial
ser = serial.Serial(port='COM4', baudrate=115200, timeout=1)
print("opened serial port")
#ser.write("zero\r".encode('utf-8'))
ms_str = "ms " + str(microsteps) + "\r"
ser.write(ms_str.encode('utf-8'))
xhat = np.matrix(np.zeros((4, 1)))
angle = 0
while abs(angle-180) > 1:
    ser.write("ga\r".encode('utf-8'))
    ser.flush()
    angle = float(readline(ser))
    print(angle)
    sleep(0.1)
u = 0
velocity = 0
accel = 0
while True:
    # read current state
    ser.write("gp\r".encode('utf-8'))
    position = readline(ser)
    
    if position == "end":
        print("endstop hit!")
        break
    # convert from steps to meters
    position = float(position)
    position = (position/microsteps)/200 * (math.pi*18.3e-3)
    print("position", position)

    ser.write("ga\r".encode('utf-8'))
    angle = readline(ser)
    if angle == "end":
        print("endstop hit!")
        break
    # convert to radian offset
    angle = float(angle)
    if abs(angle-180) > 10:
        print("angle out of range")
        break
    angle = math.pi/180 * (180-angle)
    print("angle", angle)
    
    y = np.matrix([[position], [angle]]) 
    print (y)
    xhat = (phi - K * C) * xhat + gamma * u + K * y
    print(xhat)
    u = -L * xhat
    print(u)
    
    last_velocity = velocity
    velocity = (u + m*l*0 - (M+m)*accel)/b
    accel = (velocity - last_velocity)/T
    print("velocity:", velocity)
    steps_per_sec = velocity * (microsteps*200)/(math.pi*18.3e-3)
    print("steps per sec:", int(steps_per_sec))
    ss_str = "ss " + str(int(steps_per_sec)) + "\r"
    #sio.write(ss_str.encode('utf-8'))
    sleep(T)

ser.write("stop\r".encode('utf-8'))