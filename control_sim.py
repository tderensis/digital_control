"""
Simulation functions for control system responses. Pairs best with the
plotting functions from control_plot

Requres numpy
"""
import numpy as np
import numpy.linalg as LA


def sim_regsf(phi, gamma, L, T, x0, sim_time):
    """ Simulate the response of the full state feedback regulator.
    
    Args:
        phi (matrix): The discrete plant model A matrix
        gamma (matrix): The discrete plant model B matrix
        L (matrix): The gain matrix
        T: The sampling interval in seconds
        x0 (matrix): The initial state
        sim_time: The time to simulate to in seconds
    
    Returns:
        tuple: (t, u, x) Where t is the time array, u are the inputs, and x
            are the state variables.
    """

    t = np.arange(0, sim_time, T)
    num_samples = len(t)
    (num_states, num_inputs) = gamma.shape

    x = np.matrix(np.zeros((num_states, num_samples)))
    u = np.matrix(np.zeros((num_inputs, num_samples)))
    initial = np.matrix(x0)
    if initial.shape[0] != num_states:
        x[:, 0] = np.transpose(initial)
        u[:, 0] = -L * np.transpose(initial)
    else:
        x[:, 0] = initial
        u[:, 0] = -L * initial

    for i in range(1, num_samples):
        x[:, i] = phi * x[:, i-1] + gamma * u[:, i-1]
        u[:, i] = -L * x[:, i-1]

    # Be consistent and return matrices that are the same shape np.step() would return
    x = np.transpose(x)
    u = np.transpose(u)

    return (t, u, x)
    
def sim_regob(phi, gamma, C, L, K, T, x0, sim_time):
    """ Simulate the response of the full order observer regulator.
    
    Args:
        phi (matrix): The discrete plant model A matrix
        gamma (matrix): The discrete plant model B matrix
        C (matrix): The output matrix
        L (matrix): The gain matrix
        K (matrix): The observer gain matrix
        T: The sampling interval in seconds
        x0 (matrix): The initial state
        sim_time: The time to simulate to in seconds
    
    Returns:
        tuple: (t, u, x, xhat, y) Where t is the time array, u are the inputs,
            x are the state variables, xhat are the estimated states, and y
            are the outputs.
    """

    t = np.arange(0, sim_time, T)
    num_samples = len(t)
    (num_states, num_inputs) = gamma.shape
    (num_outputs, num_states) = C.shape

    x = np.matrix(np.zeros((num_states, num_samples)))
    xhat = np.matrix(np.zeros((num_states, num_samples)))
    u = np.matrix(np.zeros((num_inputs, num_samples)))
    y = np.matrix(np.zeros((num_outputs, num_samples)))

    initial = np.matrix(x0)
    if initial.shape[0] != num_states:
        x[:, 0] = np.transpose(initial)
    else:
        x[:, 0] = initial

    y[:, 0] = C * x[:, 0]
    u[:, 0] = -L * x[:, 0]
    xhat[:, 0] = LA.lstsq(C, y[:, 0])[0]

    for i in range(1, num_samples):
        x[:, i] = phi * x[:, i-1] + gamma * u[:, i-1]
        y[:, i] = C * x[:, i] 
        xhat[:, i] = (phi - K * C) * xhat[:, i-1] + gamma * u[:, i-1] + K * y[:, i-1];
        u[:, i] = -L * xhat[:, i]

    # Be consistent and return matrices that are the same shape np.step() would return
    x = np.transpose(x)
    u = np.transpose(u)
    xhat = np.transpose(xhat)
    y = np.transpose(y)

    return (t, u, x, xhat, y)

def sim_tssf(phi, gamma, C, phia, gammaa, L1, L2, T, x0, ref, sim_time):
    """ Simulate the response of the full order observer regulator.

    Args:
        phi (matrix): The discrete plant model A matrix
        gamma (matrix): The discrete plant model B matrix
        C (matrix): The output matrix
        phia (matrix): The discrete additional dynamics A matrix
        gammaa (matrix): The discrete additional dynamics B matrix
        L1 (matrix): The state gain matrix
        L2 (matrix): The additional dynamics gain matrix
        T: The sampling interval in seconds
        x0 (matrix): The initial state
        ref (matrix): The reference input (target)
        sim_time: The time to simulate to in seconds

    Returns:
        tuple: (t, u, x, y) Where t is the time array, u are the inputs,
            x are the state variables, xhat are the estimated states, and y
            are the outputs.
    """

    t = np.arange(0, sim_time, T)
    num_samples = len(t)
    (num_states, num_inputs) = gamma.shape
    (num_outputs, num_states) = C.shape

    x = np.matrix(np.zeros((num_states, num_samples)))
    xa = np.matrix(np.zeros((gammaa.shape[0], num_samples)))
    u = np.matrix(np.zeros((num_inputs, num_samples)))
    y = np.matrix(np.zeros((num_outputs, num_samples)))

    initial = np.matrix(x0)
    if initial.shape[0] != num_states:
        x[:, 0] = np.transpose(initial)
    else:
        x[:, 0] = initial

    y[:, 0] = C * x[:, 0]
    u[:, 0] = -L1 * x[:, 0] + L2 * xa[:,0]

    for i in range(1, num_samples):
        x[:, i] = phi * x[:, i-1] + gamma * u[:, i-1]
        xa[:, i] = phia * xa[:, i-1] + gammaa * (ref[:, i-1] - y[:, i-1])
        y[:, i] = C * x[:, i]
        u[:, i] = -L1 * y[:, i] + L2 * xa[:, i]

    # Be consistent and return matrices that are the same shape np.step() would return
    x = np.transpose(x)
    u = np.transpose(u)
    y = np.transpose(y)

    return (t, u, x, y)
