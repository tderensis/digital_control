"""
Functions used to evaluate the quality of control systems. Includes measures such as
stability margins and settling time.

Requires numpy
"""
import numpy as np
from numpy import linalg as LA
import math
import cmath


def upper_gain_margin(A, B, C, discrete=True, tol=1e-3, max_gain_dB=60, output_dB=True):
    """ Calculate the upper gain margin for each input of a loop transfer function
    described by the state space matrices A, B, and C. Note that stability margins
    for MIMO systems may not represent the true robustness of the system because
    the gain or phase can change in all channels at once by a different amount.
    
    Args:
        A: The A matrix of the loop transfer function
        B: The B matrix of the loop transfer function
        C: The C matrix of the loop transfer function
        discrete (optional): True if the loop transfer function is discrete, False
            if it is continuous. Defaults to True.
        tol (optional): The tolerance to calculate the result to. Defaults to 1e-3.
        max_gain_dB (optional): The maximum dB to search to. Defaults to 60 dB
        ouput_dB (optional): True if the output should be in dB, False if the result
            should be returned as gain. Defaults to True.
    
    Returns:
        list: The list of upper gain margins at each input. Units dependent on the value of output_dB.
    """
    
    (n, p) = B.shape
    max_gain = max(1, math.pow(10, max_gain_dB/20))
    gain_list = [None] * p
    
    # Create a measure of stability for the poles based on if the system is discrete
    if discrete is True:
        # Stable poles for discrete systems are inside the unit circle
        def is_unstable(poles):
            return max(abs(poles)) > 1
    else:
        # Stable poles for continuous systems are negative
        def is_unstable(poles):
            return max([ pole.real for pole in poles ]) > 0

    for i in range(0, p):
        # Use the bisect method for calculating the gain margin
        t1 = 1
        t2 = max_gain
        gain_mat = np.eye(p)
        gain = t1

        while 20 * math.log(t2/t1, 10) > tol:
            gain = (t1 + t2)/2;
            
            # Multiply the current input by the gain
            gain_mat[i, i] = gain
            eig_vals, v = LA.eig(A - B*gain_mat*C)
            
            if is_unstable(eig_vals):
                t2 = gain # decrease the gain
            else:
                t1 = gain # increase the gain
        
        if output_dB is True:
            gain_list[i] = 20 * math.log(gain, 10)
        else:
            gain_list[i] = gain
    
    return gain_list


def lower_gain_margin(A, B, C, discrete=True, tol=1e-3, min_gain_dB=-60, output_dB=True):
    """ Calculate the lower gain margin for each input of a loop transfer function
    described by the state space matrices A, B, and C. Note that stability margins
    for MIMO systems may not represent the true robustness of the system because
    the gain or phase can change in all channels at once by a different amount.
    Not all systems have lower gain margin. These systems will report the minimum value.
    
    Args:
        A: The A matrix of the loop transfer function
        B: The B matrix of the loop transfer function
        C: The C matrix of the loop transfer function
        discrete (optional): True if the loop transfer function is discrete, False
            if it is continuous. Defaults to True.
        tol (optional): The tolerance to calculate the result to. Defaults to 1e-3.
        min_gain_dB (optional): The minimum dB to search to. Defaults to -60 dB
        ouput_dB (optional): True if the output should be in dB, False if the result
            should be returned as gain. Defaults to True.
    
    Returns:
        list: The list of lower gain margins for each input. Units dependent on the value of output_dB.
    """
    
    (n, p) = B.shape
    min_gain = min(1, math.pow(10, min_gain_dB/20))
    gain_list = [None] * p

    # Create a measure of stability for the poles based on if the system is discrete
    if discrete is True:
        # Stable poles for discrete systems are inside the unit circle
        def is_unstable(poles):
            return max(abs(poles)) > 1
    else:
        # Stable poles for continuous systems are negative
        def is_unstable(poles):
            return max([ pole.real for pole in poles ]) > 0

    for i in range(0, p):
        # Use the bisect method for calculating the gain margin
        t1 = min_gain
        t2 = 1
        gain_mat = np.eye(p)
        gain = t1

        while 20 * math.log(t2/t1, 10) > tol:
            gain = (t1 + t2)/2;

            # Multiply the current input by the gain
            gain_mat[i, i] = gain
            eig_vals, v = LA.eig(A - B*gain_mat*C)
            
            if is_unstable(eig_vals):
                t1 = gain # increase the gain
            else:
                t2 = gain # decrease the gain
        
        if output_dB is True:
            gain_list[i] = 20 * math.log(gain, 10)
        else:
            gain_list[i] = gain
    
    return gain_list


def phase_margin(A, B, C, discrete=True, tol=1e-3, max_angle_deg=120):
    """ Calculate the phase margin for each input of a loop transfer function
    described by the state space matrices A, B, and C. Note that stability margins
    for MIMO systems may not represent the true robustness of the system because
    the gain or phase can change in all channels at once by a different amount.
    
    Args:
        A: The A matrix of the loop transfer function
        B: The B matrix of the loop transfer function
        C: The C matrix of the loop transfer function
        discrete (optional): True if the loop transfer function is discrete, False
            if it is continuous. Defaults to True.
        tol (optional): The tolerance to calculate the result to. Defaults to 1e-3.
        max_angle_deg (optional): The maximum angle to search to. Defaults to 120 degrees
    
    Returns:
        list: The list of phase margins for each input. Units are degrees.
    """
    
    (n, p) = B.shape
    max_angle = max(1, max_angle_deg)
    angle_list = [None] * p
    
    # Create a measure of stability for the poles based on if the system is discrete
    if discrete is True:
        # Stable poles for discrete systems are inside the unit circle
        def is_stable(poles):
            return max(abs(poles)) <= 1
    else:
        # Stable poles for continuous systems are negative
        def is_stable(poles):
            return max([ pole.real for pole in poles ]) <= 0

    for i in range(0, p):
        # Use the bisect method for calculating the phase margin
        t1 = 1
        t2 = max_angle
        gain_mat = np.eye(p, dtype=complex)
        angle = t1

        while t2 - t1 > tol:
            angle = (t1 + t2)/2;
            
            # Multiply the current input by the phase offset
            gain_mat[i, i] = cmath.exp(-1j * angle * math.pi/180)
            eig_vals, v = LA.eig(A - B*gain_mat*C)
            
            if is_stable(eig_vals):
                t1 = angle # increase the angle
            else:
                t2 = angle # decrease the angle
        
        angle_list[i] = angle
    
    return angle_list


def print_stability_margins(A, B, C, discrete=True, tol=1e-3):
    """ Print the stability margins (gain and phase) for each input of a loop
    transfer function described by the state space matrices A, B, and C.
    Note that stability margins for MIMO systems may not represent the true
    robustness of the system because the gain or phase can change in all channels
    at once by a different amount.
    
    Args:
        A: The A matrix of the loop transfer function
        B: The B matrix of the loop transfer function
        C: The C matrix of the loop transfer function
        discrete (optional): True if the loop transfer function is discrete, False
            if it is continuous. Defaults to True.
        tol (optional): The tolerance to calculate the result to. Defaults to 1e-3.
    
    Returns:
        Nothing
    """

    ugm = upper_gain_margin(A, B, C, discrete=discrete, tol=tol)
    lgm = lower_gain_margin(A, B, C, discrete=discrete, tol=tol)
    phm = phase_margin(A, B, C, discrete=discrete, tol=tol)

    for i in range(1, len(ugm)+1):
        print("Input " + str(i) + " upper gain margin = " + str(round(ugm[i-1], 2)) + " dB")
        print("Input " + str(i) + " lower gain margin = " + str(round(lgm[i-1], 2)) + " dB")
        print("Input " + str(i) + " phase margin = " + str(round(phm[i-1],2)) + " deg")


def settling_time(t, y, percent=0.02, start=None, end=None):
    """ Calculate the time it takes for each output to reach its
    final value to within a given percentage.
    
    Args:
        t (array): The time points (1 x n)
        y (ndarray): A list of the output vectors (n, m), where m is the number of states.
        percent (optional): The percent to which the output needs to settle to.
        start (optional): The starting value to use for calculations. If none is given, then
            the max of the first values of y are used. Default is None.
        end (optional): The end value to use for calculations. If none is given, then
            the min of the last values of y are used.
        
    Returns:
        The settling time in seconds.
    """
    
    settling_times = []
    (num_samples, states) = y.shape
    
    if start is None:
        start = max([abs(n) for n in y[0,:].tolist()[0]])

    if end is None:
        end = round(min([abs(n) for n in y[-1,:].tolist()[0]]), 3)

    yout = np.transpose(y).tolist()
    limit = percent * abs(start - end)
    limit_high = end + limit
    limit_low = end - limit

    for state in range(0, states):
        i = num_samples
        for y in reversed(yout[state]):
            i -= 1
            if y > limit_high or y < limit_low:
                settling_times.append(t[i])
                break

    return max(settling_times)

    
def ltf_regsf(sys_ol, L):
    """ Construct the the loop transfer function of the full state feedback
    regulator system. Used for calculating stability.
    
    Args:
        sys_ol (StateSpace): The state-space model of the plant
        L (matrix): The gain matrix
        
    Returns:
        tuple: (A, B, C) Where A, B, and C are the matrices that describe the
            loop transfer function
    """

    A = sys_ol.A
    B = sys_ol.B
    return (A, B, L)


def ltf_regob(sys_ol, L, K):
    """ Construct the the loop transfer function of the full order
    observer system. Used for calculating stability.
    
    Args:
        sys_ol (StateSpace): The state-space model of the plant
        L (matrix): The gain matrix
        K (matrix): The observer gain matrix
        
    Returns:
        tuple: (A, B, C) Where A, B, and C are the matrices that describe the
            loop transfer function
    """

    A = sys_ol.A
    B = sys_ol.B
    C = sys_ol.C
    (n, p) = B.shape
    A_ltf_top_row = np.concatenate((A, np.zeros((n, n))), axis=1)
    A_ltf_bot_row = np.concatenate((K * C, A - (K * C) - (B * L)), axis=1)
    A_ltf = np.concatenate((A_ltf_top_row, A_ltf_bot_row), axis=0)
    B_ltf = np.concatenate((B, np.zeros((n, p))), axis=0)
    C_ltf = np.concatenate((np.zeros((p, n)), L), axis=1)

    return (A_ltf, B_ltf, C_ltf)


def ltf_tsob(sys_ol, Aa, Ba, L1, L2, K):
    """ Construct the the loop transfer function of the full order
    observer tracking system. Used for calculating stability.
    
    Args:
        sys_ol (StateSpace): The state-space model of the plant
        Aa (matrix): The additional dynamics state matrix
        Ba (matrix): The additional dynamics input matrix
        L1 (matrix): The plant gain matrix
        L2 (matrix): The additional dynamics gain matrix
        K (matrix): The observer gain matrix
        
    Returns:
        tuple: (A, B, C) Where A, B, and C are the matrices that describe the
            loop transfer function
    """

    A = sys_ol.A
    B = sys_ol.B
    C = sys_ol.C
    (n, p) = B.shape
    (na, pa) = Ba.shape
    A_ltf_top_row = np.concatenate((A, np.zeros((n, n+na))), axis=1)
    A_ltf_mid_row = np.concatenate((K * C, A - K * C - B * L1, -B * L2), axis=1)
    A_ltf_bot_row = np.concatenate((Ba * C, np.zeros((na, n)), Aa), axis=1)
    A_ltf = np.concatenate((A_ltf_top_row, A_ltf_mid_row, A_ltf_bot_row), axis=0)
    B_ltf = np.concatenate((B, np.zeros((n + na, pa))), axis=0)
    C_ltf = np.concatenate((np.zeros((p, n)), L1, L2), axis=1)

    return (A_ltf, B_ltf, C_ltf)


def ltf_tssf(sys_ol, Aa, Ba, L1, L2):
    """ Construct the the loop transfer function of the full state
    feedback tracking system. Used for calculating stability.
    
    Args:
        sys_ol (StateSpace): The state-space model of the plant
        Aa (matrix): The additional dynamics state matrix
        Ba (matrix): The additional dynamics input matrix
        L1 (matrix): The plant gain matrix
        L2 (matrix): The additional dynamics gain matrix
        
    Returns:
        tuple: (A, B, C) Where A, B, and C are the matrices that describe the
            loop transfer function
    """

    A = sys_ol.A
    B = sys_ol.B
    C = sys_ol.C
    (n, p) = B.shape
    (na, pa) = Ba.shape
    A_ltf_top_row = np.concatenate((A, np.zeros((n, na))), axis=1)
    A_ltf_bot_row = np.concatenate((Ba * C, Aa), axis=1)
    A_ltf = np.concatenate((A_ltf_top_row, A_ltf_bot_row), axis=0)
    B_ltf = np.concatenate((B, np.zeros((na, pa))), axis=0)
    C_ltf = np.concatenate((L1, L2), axis=1)
    
    return (A_ltf, B_ltf, C_ltf)
