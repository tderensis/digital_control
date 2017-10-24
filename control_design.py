"""
Functions that construct stable closed loop control systems. Many of the
methods here are adapted from Digital Control: A State-Space Approach and
accompanying courses at URI.

Requires numpy, scipy, control
"""
from __future__ import print_function
import control_poles
import control
import numpy as np
from numpy import linalg as LA
from scipy import signal
import cmath


def design_regsf(sys_c_ol, sampling_interval, desired_settling_time, spoles=None):
    """ Design a digital full state feedback regulator with the desired settling time.
    
    Args:
        sys_c_ol (StateSpace): The continouous plant model
        sampling_interval: The sampling interval for the digital control system in seconds.
        desired_settling_time: The desired settling time in seconds
        spoles (optional): The desired closed loop poles. If not supplied, then optimal
            poles will try to be used. Default is None.
            
    Returns:
        tuple: (sys_d_ol, L) Where sys_d_ol is the discrete plant and L is the stablizing
            gain matrix.
    """

    # Make sure the system is in fact continuous and not discrete
    if sys_c_ol.dt != None:
        print("Error: Function expects continuous plant")
        return None

    A = sys_c_ol.A
    B = sys_c_ol.B
    C = sys_c_ol.C
    D = sys_c_ol.D
    
    num_states = A.shape[0]
    num_inputs = B.shape[1]
    num_outputs = C.shape[0]
    
    # Convert to discrete system using zero order hold method
    sys_d_ol = sys_c_ol.to_discrete(sampling_interval, method="zoh")
    phi = sys_d_ol.A
    gamma = sys_d_ol.B

    # Check controlability of the discrete system
    controllability_mat = control.ctrb(phi, gamma)
    rank = LA.matrix_rank(controllability_mat)
    if rank != num_states:
        print("Error: System is not controlable")
        return None

    # Check observability of the discrete system
    observability_mat = control.obsv(phi, C)
    rank = LA.matrix_rank(observability_mat)
    if rank != num_states:
        print("Error: System is not observable")
        return None
    
    # Choose poles if none were given
    
    if spoles == None:
        spoles = []
        
        (sys_spoles, vectors) = LA.eig(A)
        
        # first go through the system poles and see if they are suitable.
        s1_normalized = control_poles.bessel_spoles(1, desired_settling_time)[0]
        for pole in sys_spoles:
            if pole.real < s1_normalized:
                # Use sufficiently damped plant poles: plant poles whose real parts lie to the left of s1/Ts.
                spoles.extend([pole])
            elif pole.imag != 0 and pole.real > s1_normalized and pole.real < 0:
                # Replace real part of a complex pole that is not sufficiently damped with s1/Ts
                spoles.extend([complex(s1_normalized, pole.imag)])
            elif pole.real > 0 and pole.real > s1_normalized:
                # Reflect the pole about the imaginary axis and use that
                spoles.extend([complex(-pole.real, pole.imag)])
        
        num_spoles_left = num_states - len(spoles)
        if num_spoles_left > 0:
            # Use normalized bessel poles for the rest
            spoles.extend(control_poles.bessel_spoles(num_spoles_left, desired_settling_time))
    
    zpoles = control_poles.spoles_to_zpoles(spoles, sampling_interval)

    # place the poles such that ...
    full_state_feedback = signal.place_poles(phi, gamma, zpoles)
    
    # Check the poles for stability
    for zpole in full_state_feedback.computed_poles:
        if abs(zpole) >= 1:
            print("Computed pole is not stable")
            return None
    
    L = full_state_feedback.gain_matrix

    return (sys_d_ol, np.matrix(L))


def design_regob(sys_c_ol, sampling_interval, desired_settling_time,
                 desired_observer_settling_time=None, spoles=None, sopoles=None):
    """ Design a digital full order observer regulator with the desired settling time.
    
    Args:
        sys_c_ol (StateSpace): The continouous plant model
        sampling_interval: The sampling interval for the digital control system in seconds.
        desired_settling_time: The desired settling time in seconds
        desired_observer_settling_time (optional): The desired observer settling time
            in seconds. If not provided the observer settling time will be 4 times faster
            than the overall settling time. Default is None.
        spoles (optional): The desired closed loop poles. If not supplied, then optimal
            poles will try to be used. Default is None.
        sopoles (optional): The desired observer poles. If not supplied, then optimal
            poles will try to be used. Default is None.
    
    Returns:
        tuple: (sys_d_ol, L, K) Where sys_d_ol is the discrete plant, L is the stablizing
            gain matrix, and K is the observer gain matrix.
    """

    # Make sure the system is in fact continuous and not discrete
    if sys_c_ol.dt != None:
        print("Error: Function expects continuous plant")
        return None

    A = sys_c_ol.A
    B = sys_c_ol.B
    C = sys_c_ol.C
    D = sys_c_ol.D
    
    num_states = A.shape[0]
    num_inputs = B.shape[1]
    num_outputs = C.shape[0]
    
    # Convert to discrete system using zero order hold method
    sys_d_ol = sys_c_ol.to_discrete(sampling_interval, method="zoh")
    phi = sys_d_ol.A
    gamma = sys_d_ol.B

    # Check controlability of the discrete system
    controllability_mat = control.ctrb(phi, gamma)
    rank = LA.matrix_rank(controllability_mat)
    if rank != num_states:
        print(rank, num_states)
        print("Error: System is not controlable")
        return None

    # Check observability of the discrete system
    observability_mat = control.obsv(phi, C)
    rank = LA.matrix_rank(observability_mat)
    if rank != num_states:
        print("Error: System is not observable")
        return None
    
    # Choose poles if none were given
    
    if spoles == None:
        spoles = []
        
        (sys_spoles, vectors) = LA.eig(A)
        
        # first go through the system poles and see if they are suitable.
        s1_normalized = control_poles.bessel_spoles(1, desired_settling_time)[0]
        for pole in sys_spoles:
            if pole.real < s1_normalized:
                # Use sufficiently damped plant poles: plant poles whose real parts lie to the left of s1/Ts.
                spoles.extend([pole])
            elif pole.imag != 0 and pole.real > s1_normalized and pole.real < 0:
                # Replace real part of a complex pole that is not sufficiently damped with s1/Ts
                pole = [complex(s1_normalized, pole.imag)]
                spoles.extend(pole)
            elif pole.real > 0 and pole.real > s1_normalized:
                # Reflect the pole about the imaginary axis and use that
                pole = [complex(-pole.real, pole.imag)]
                spoles.extend(pole)
        
        num_spoles_left = num_states - len(spoles)
        if num_spoles_left > 0:
            # Use normalized bessel poles for the rest
            spoles.extend(control_poles.bessel_spoles(num_spoles_left, desired_settling_time))
    
    zpoles = control_poles.spoles_to_zpoles(spoles, sampling_interval)

    # place the poles such that eig(phi - gamma*L) are inside the unit circle
    full_state_feedback = signal.place_poles(phi, gamma, zpoles)
    
    # Check the poles for stability just in case
    for zpole in full_state_feedback.computed_poles:
        if abs(zpole) >= 1:
            print("Computed pole is not stable")
            return None
    
    L = full_state_feedback.gain_matrix

    # Choose poles if none were given
    if sopoles == None:
        sopoles = []
        if desired_observer_settling_time == None:
            desired_observer_settling_time = desired_settling_time/4
        
        # TODO: Find existing poles based on the rules. For now just use bessel
        
        num_sopoles_left = num_states - len(sopoles)
        
        if num_sopoles_left > 0:
            # Use normalized bessel poles for the rest
            sopoles.extend(control_poles.bessel_spoles(num_sopoles_left, desired_settling_time))
    
    zopoles = control_poles.spoles_to_zpoles(sopoles, sampling_interval)
        
    # Find K such that eig(phi - KC) are inside the unit circle
    full_state_feedback = signal.place_poles(np.transpose(phi), np.transpose(C), zopoles)
    
    # Check the poles for stability just in case
    for zopole in full_state_feedback.computed_poles:
        if abs(zopole) >= 1:
            print("Computed observer pole is not stable")
            return None
    
    K = np.transpose(full_state_feedback.gain_matrix)

    return (sys_d_ol, np.matrix(L), np.matrix(K))


def design_tsob(sys_c_ol, sampling_interval, desired_settling_time,
                desired_observer_settling_time=None, spoles=None, sopoles=None):
    """ Design a digital full order observer tracking system with the desired settling time.
    
    Args:
        sys_c_ol (StateSpace): The continouous plant model
        sampling_interval: The sampling interval for the digital control system in seconds.
        desired_settling_time: The desired settling time in seconds
        desired_observer_settling_time (optional): The desired observer settling time
            in seconds. If not provided the observer settling time will be 4 times faster
            than the overall settling time. Default is None.
        spoles (optional): The desired closed loop poles. If not supplied, then optimal
            poles will try to be used. Default is None.
        sopoles (optional): The desired observer poles. If not supplied, then optimal
            poles will try to be used. Default is None.
            
    Returns:
        tuple: (sys_d_ol, phia, gammaa, L1, L2, K) Where sys_d_ol is the discrete plant,
            phia is the discrete additional dynamics A matrix, gammaa is the discrete
            additional dynamics B matrix, L1 is the plant gain matrix, L2 is the
            additional gain matrix, and K is the observer gain matrix.
    """

    # Make sure the system is in fact continuous and not discrete
    if sys_c_ol.dt != None:
        print("Error: Function expects continuous plant")
        return None

    A = sys_c_ol.A
    B = sys_c_ol.B
    C = sys_c_ol.C
    D = sys_c_ol.D
    
    num_states = A.shape[0]
    num_inputs = B.shape[1]
    num_outputs = C.shape[0]
    
    # Convert to discrete system using zero order hold method
    sys_d_ol = sys_c_ol.to_discrete(sampling_interval, method="zoh")
    phi = sys_d_ol.A
    gamma = sys_d_ol.B
    
    # Check controlability of the discrete system
    controllability_mat = control.ctrb(phi, gamma)
    rank = LA.matrix_rank(controllability_mat)
    if rank != num_states:
        print(rank, num_states)
        print("Error: System is not controlable")
        return None

    # Check observability of the discrete system
    observability_mat = control.obsv(phi, C)
    rank = LA.matrix_rank(observability_mat)
    if rank != num_states:
        print("Error: System is not observable")
        return None
    
    # Create the design model with additional dynamics
    phia = np.eye(num_inputs)
    gammaa = np.zeros((num_inputs, 1))
    gammaa[0,0] = 1
    phid_top_row = np.concatenate((phi, np.zeros((num_states, num_inputs))), axis=1)
    phid_bot_row = np.concatenate((gammaa*C, phia), axis=1)
    phid = np.concatenate((phid_top_row, phid_bot_row), axis=0)
    gammad = np.concatenate((gamma, np.zeros((num_inputs, num_inputs))), axis=0)

    # Choose poles if none were given
    
    if spoles == None:
        spoles = []
        
        (sys_spoles, vectors) = LA.eig(A)
        
        # first go through the system poles and see if they are suitable.
        s1_normalized = control_poles.bessel_spoles(1, desired_settling_time)[0]

        for pole in sys_spoles:
            if pole.real < s1_normalized:
                # Use sufficiently damped plant poles: plant poles whose real parts lie to the left of s1/Ts.
                spoles.extend([pole])
            elif pole.imag != 0 and pole.real > s1_normalized and pole.real < 0:
                # Replace real part of a complex pole that is not sufficiently damped with s1/Ts
                pole = [complex(s1_normalized, pole.imag)]
                spoles.extend(pole)
            elif pole.real > 0 and pole.real > s1_normalized:
                # Reflect the pole about the imaginary axis and use that
                pole = [complex(-pole.real, pole.imag)]
                spoles.extend(pole)
        
        num_spoles_left = phid.shape[0] - len(spoles)

        if num_spoles_left > 0:
            # Use normalized bessel poles for the rest
            spoles.extend(control_poles.bessel_spoles(num_spoles_left, desired_settling_time))

    zpoles = control_poles.spoles_to_zpoles(spoles, sampling_interval)

    # place the poles such that eig(phi - gamma*L) are inside the unit circle
    full_state_feedback = signal.place_poles(phid, gammad, zpoles)
    
    # Check the poles for stability just in case
    for zpole in full_state_feedback.computed_poles:
        if abs(zpole) >= 1:
            print("Computed pole is not stable")
            return None
    
    L = full_state_feedback.gain_matrix
    L1 = L[:,0:num_states]
    L2 = L[:,num_states:]

    # Choose poles if none were given
    if sopoles == None:
        sopoles = []
        if desired_observer_settling_time == None:
            desired_observer_settling_time = desired_settling_time/4
        
        # TODO: Find existing poles based on the rules. For now just use bessel
        
        num_sopoles_left = num_states - len(sopoles)
        
        if num_sopoles_left > 0:
            # Use normalized bessel poles for the rest
            sopoles.extend(control_poles.bessel_spoles(num_sopoles_left, desired_settling_time))
    
    zopoles = control_poles.spoles_to_zpoles(sopoles, sampling_interval)
        
    # Find K such that eig(phi - KC) are inside the unit circle
    full_state_feedback = signal.place_poles(np.transpose(phi), np.transpose(C), zopoles)
    
    # Check the poles for stability just in case
    for zopole in full_state_feedback.computed_poles:
        if abs(zopole) >= 1:
            print("Computed observer pole is not stable")
            return None
    
    K = np.transpose(full_state_feedback.gain_matrix)

    return (sys_d_ol, phia, gammaa, np.matrix(L1), np.matrix(L2), np.matrix(K))
