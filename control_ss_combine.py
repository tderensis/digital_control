"""
Functions used to combine and connect state space systems. Based on
the paper "Combining and Connection Linear, Multi-Input, Multi-Output
Subsystem Models" by Eugene L. Duke.

Requres numpy
"""
import numpy as np
from scipy import signal
import scipy.linalg as LA


def ss_parallel(sys1, sys2, F1=None, F2=None):
    """ Combine the two independent parallel state space systems into
    a single combined system.
    
     w1   ----   u1   ------   y1
    ---> | F1 | ---> | sys1 | --->
          ----        ------
          
     w2   ----   u2   ------   y2
    ---> | F2 | ---> | sys2 | --->
          ----        ------

    Args:
        sys1 (StateSpace): The first state space system
        sys2 (StateSpace): The second state space system
        F1 (matrix, optional): The first system's input selection matrix.
            Defaults to the identity matrix
        F2 (matrix, optional): The second system's input selection matrix.
            Defaults to the identity matrix
    
    Returns:
        StateSpace: The parallel system where 
                [ x1 ]      [ w1 ]      [ y1 ]
            x = [----], u = [----], y = [----]
                [ x2 ]      [ w2 ]      [ y2 ]
    """
    
    A1 = sys1.A
    B1 = sys1.B
    C1 = sys1.C
    D1 = sys1.D
    A2 = sys2.A
    B2 = sys2.B
    C2 = sys2.C
    D2 = sys2.D
    
    (n1, p1) = B1.shape
    (q1, n1) = C1.shape
    (n2, p2) = B2.shape
    (q2, n2) = C2.shape
    
    if F1 is None:
        F1 = np.matrix(np.eye(p1))
        
    if F2 is None:
        F2 = np.matrix(np.eye(p2))
    
    A_top_row = np.concatenate((A1, np.zeros((n1, n2))), axis=1)
    A_bot_row = np.concatenate((np.zeros((n2, n1)), A2), axis=1)
    A = np.concatenate((A_top_row, A_bot_row), axis=0)
    
    B_top_row = np.concatenate((B1*F1, np.zeros((p1, n2))), axis=1)
    B_bot_row = np.concatenate((np.zeros((p2, n1)), B2*F2), axis=1)
    B = np.concatenate((B_top_row, B_bot_row), axis=0)
    
    C_top_row = np.concatenate((C1, np.zeros((q1, n2))), axis=1)
    C_bot_row = np.concatenate((np.zeros((q2, n1)), sys), axis=1)
    C = np.concatenate((C_top_row, C_bot_row), axis=0)
    
    D_top_row = np.concatenate((D1*F1, np.zeros((q1, n2))), axis=1)
    D_bot_row = np.concatenate((np.zeros((q2, n1)), D2*F2), axis=1)
    D = np.concatenate((D_top_row, D_bot_row), axis=0)
    
    return signal.StateSpace(A, B, C, D)


def ss_sum(sys1, sys2, F1=None, F2=None, H1=None, H2=None):
    """ Sum two systems which have independent input vectors.
    
     w1   ----   u1   ------   y1   ----   z1
    ---> | F1 | ---> | sys1 | ---> | H1 | ---|
          ----        ------        ----     |
                                           + v  y
                                            (x) --->
                                           + ^
     w2   ----   u2   ------   y2   ----     |
    ---> | F2 | ---> | sys2 | ---> | H2 | ---|
          ----        ------        ----   z2

    Args:
        sys1 (StateSpace): The first state space system
        sys2 (StateSpace): The second state space system
        F1 (matrix, optional): The first system's input selection matrix.
            Defaults to the identity matrix
        F2 (matrix, optional): The second system's input selection matrix.
            Defaults to the identity matrix
        H1 (matrix, optional): The first system's output selection matrix.
            Defaults to the identity matrix
        H2 (matrix, optional): The second system's output selection matrix.
            Defaults to the identity matrix
    
    Returns:
        StateSpace: The summed system where 
                [ x1 ]      [ w1 ]
            x = [----], u = [----], y = z1 + z2
                [ x2 ]      [ w2 ]
    """
    
    A1 = sys1.A
    B1 = sys1.B
    C1 = sys1.C
    D1 = sys1.D
    A2 = sys2.A
    B2 = sys2.B
    C2 = sys2.C
    D2 = sys2.D
    
    (n1, p1) = B1.shape
    (q1, n1) = C1.shape
    (n2, p2) = B2.shape
    (q2, n2) = C2.shape
    
    if F1 is None:
        F1 = np.matrix(np.eye(p1))
        
    if F2 is None:
        F2 = np.matrix(np.eye(p2))
    
    if H1 is None:
        H1 = np.matrix(np.eye(q1))
        
    if H2 is None:
        H2 = np.matrix(np.eye(q2))
 
    A_top_row = np.concatenate((A1, np.zeros((n1, n2))), axis=1)
    A_bot_row = np.concatenate((np.zeros((n2, n1)), A2), axis=1)
    A = np.concatenate((A_top_row, A_bot_row), axis=0)
    
    B_top_row = np.concatenate((B1*F1, np.zeros((p1, n2))), axis=1)
    B_bot_row = np.concatenate((np.zeros((p2, n1)), B2*F2), axis=1)
    B = np.concatenate((B_top_row, B_bot_row), axis=0)

    C = np.concatenate((H1*C1, H2*C2), axis=1)

    D = np.concatenate((H1*D1*F1, H2*D2*F2), axis=1)
    
    return signal.StateSpace(A, B, C, D)
    
    
def ss_concatenate(sys1, sys2, F1=None, H1=None):
    """ Concatenate two systems in series.
    
     w    ----   u1   ------   y1   ----   u2   ------   y2
    ---> | F1 | ---> | sys1 | ---> | H1 | ---> | sys2 | --->
          ----        ------        ----        ------
    

    Args:
        sys1 (StateSpace): The first state space system
        sys2 (StateSpace): The second state space system
        F1 (matrix, optional): The first system's input selection matrix.
            Defaults to the identity matrix
        H1 (matrix, optional): The first system's output selection matrix.
            Defaults to the identity matrix
    
    Returns:
        StateSpace: The series system where 
                [ x1 ]
            x = [----], u = w, y = y2
                [ x2 ]
    """

    A1 = sys1.A
    B1 = sys1.B
    C1 = sys1.C
    D1 = sys1.D
    A2 = sys2.A
    B2 = sys2.B
    C2 = sys2.C
    D2 = sys2.D
    
    (n1, p1) = B1.shape
    (q1, n1) = C1.shape
    (n2, p2) = B2.shape
    (q2, n2) = C2.shape
    
    if F1 is None:
        F1 = np.matrix(np.eye(p1))
    
    if F2 is None:
        F2 = np.matrix(np.eye(p2))
    
    if H1 is None:
        H1 = np.matrix(np.eye(q1))
    
    if H2 is None:
        H2 = np.matrix(np.eye(q2))

    A_top_row = np.concatenate((A1, np.zeros((n1, n2))), axis=1)
    A_bot_row = np.concatenate(B2*H1*C1, A2), axis=1)
    A = np.concatenate((A_top_row, A_bot_row), axis=0)

    B = np.concatenate((B1*F1, B2*H1*D1*F1), axis=0)

    C = np.concatenate((D2*H1*C1, C2), axis=1)

    D = np.matrix(D2*H1*D1*F1)
    
    return signal.StateSpace(A, B, C, D)


def ss_feedback(sys1, sys2, F1=None, F2=None, H2=None):
    """ Feedback connection of two systems.
    
     w    ----   u +         u1        ------           y1
    ---> | F1 | --->(x)-------------> | sys1 | -------------------->
          ----       ^ -               ------                  |
                     |                                         |
                     | z2  ----   y2   ------   u2   ----      |
                     |--- | H2 | <--- | sys2 | <--- | F2 | <---|
                           ----        ------        ----

    Args:
        sys1 (StateSpace): The first state space system
        sys2 (StateSpace): The second state space system
        F1 (matrix, optional): The first system's input selection matrix.
            Defaults to the identity matrix
        F2 (matrix, optional): The second system's input selection matrix.
            Defaults to the identity matrix
        H2 (matrix, optional): The second system's output selection matrix.
            Defaults to the identity matrix
    
    Returns:
        StateSpace: The feedback system where 
                [ x1 ]
            x = [----], u = w, y = y1
                [ x2 ]
    """
    
    A1 = sys1.A
    B1 = sys1.B
    C1 = sys1.C
    D1 = sys1.D
    A2 = sys2.A
    B2 = sys2.B
    C2 = sys2.C
    D2 = sys2.D
    
    (n1, p1) = B1.shape
    (q1, n1) = C1.shape
    (n2, p2) = B2.shape
    (q2, n2) = C2.shape
    
    if F1 is None:
        F1 = np.matrix(np.eye(p1))
        
    if F2 is None:
        F2 = np.matrix(np.eye(p2))
    
    if H2 is None:
        H2 = np.matrix(np.eye(q2))
    
    I = np.matrix(np.eye(q1))
    N = LA.inv(I + D1*H2*D2*F2)

    A_top_row = np.concatenate((A1 - B1*H2*D2*F2*N*C1, -B1*H2*(C2 - D2*F2*N*D1*H2*C2)), axis=1)
    A_bot_row = np.concatenate((B2*F2*N*C1, A2 - B2*F2*N*D1*H2*C2), axis=1)
    A = np.concatenate((A_top_row, A_bot_row), axis=0)
    
    B_top_row = np.concatenate((B1*F1 - B1, np.zeros((p1, n2))), axis=1)
    B_bot_row = np.concatenate((np.zeros((p2, n1)), B2*F2), axis=1)
    B = np.concatenate((B_top_row, B_bot_row), axis=0)
    
    C = np.concatenate((N*C1, -N*D1*H2*C2), axis=1)

    D = np.matrix(N*D1*F1)
    
    return signal.StateSpace(A, B, C, D)


def ss_common_in(sys1, sys2, F1=None):
    """ Connection of two systems with a common input
    
                          ------   y1
                   |---> | sys1 | --->
                   |      ------
     w    ----   u |
    ---> | F1 | ---|
          ----     |
                   |      ------   y2
                   |---> | sys2 | --->
                          ------

    Args:
        sys1 (StateSpace): The first state space system
        sys2 (StateSpace): The second state space system
        F1 (matrix, optional): Each system's input selection matrix.
            Defaults to the identity matrix
    
    Returns:
        StateSpace: The common input system where 
                [ x1 ]             [ y1 ]
            x = [----], u = w, y = [----]
                [ x2 ]             [ y2 ]
    """
    
    A1 = sys1.A
    B1 = sys1.B
    C1 = sys1.C
    D1 = sys1.D
    A2 = sys2.A
    B2 = sys2.B
    C2 = sys2.C
    D2 = sys2.D
    
    (n1, p1) = B1.shape
    (q1, n1) = C1.shape
    (n2, p2) = B2.shape
    (q2, n2) = C2.shape
    
    if F1 is None:
        F1 = np.matrix(np.eye(p1))
    
    A_top_row = np.concatenate((A1, np.zeros((n1, n2))), axis=1)
    A_bot_row = np.concatenate((np.zeros((n2, n1)), A2), axis=1)
    A = np.concatenate((A_top_row, A_bot_row), axis=0)
    
    B = np.concatenate((B1*F1, B2*F1), axis=0)
    
    C_top_row = np.concatenate((C1, np.zeros((q1, n2))), axis=1)
    C_bot_row = np.concatenate((np.zeros((q2, n1)), sys), axis=1)
    C = np.concatenate((C_top_row, C_bot_row), axis=0)

    D = np.concatenate((D1*F1, D2*F1), axis=0)
    
    return signal.StateSpace(A, B, C, D)


 def ss_sum_common_in(sys1, sys2, F1=None, H1=None, H2=None):
    """ Connection of two systems with a common input
    
                          ------   y1   ----   z1
                   |---> | sys1 | ---> | H1 | ---|
                   |      ------        ----     |
     w    ----   u |                           + v   y
    ---> | F1 | ---|                            (x) --->
          ----     |                           + ^
                   |      ------   y2   ----     |
                   |---> | sys2 | ---> | H2 | ---|
                          ------        ----   z2

    Args:
        sys1 (StateSpace): The first state space system
        sys2 (StateSpace): The second state space system
        F1 (matrix, optional): The first system's input selection matrix.
            Defaults to the identity matrix
        H1 (matrix, optional): The first system's output selection matrix.
            Defaults to the identity matrix
        H2 (matrix, optional): The second system's output selection matrix.
            Defaults to the identity matrix
    
    Returns:
        StateSpace: The summation of common input system where 
                [ x1 ]
            x = [----], u = w, y = z1 + z2
                [ x2 ]
    """
    
    A1 = sys1.A
    B1 = sys1.B
    C1 = sys1.C
    D1 = sys1.D
    A2 = sys2.A
    B2 = sys2.B
    C2 = sys2.C
    D2 = sys2.D
    
    (n1, p1) = B1.shape
    (q1, n1) = C1.shape
    (n2, p2) = B2.shape
    (q2, n2) = C2.shape
    
    if F1 is None:
        F1 = np.matrix(np.eye(p1))
    
    if H1 is None:
        H1 = np.matrix(np.eye(q1))
    
    if H2 is None:
        H2 = np.matrix(np.eye(q2))

    A_top_row = np.concatenate((A1, np.zeros((n1, n2))), axis=1)
    A_bot_row = np.concatenate(B2*H1*C1, A2), axis=1)
    A = np.concatenate((A_top_row, A_bot_row), axis=0)

    B = np.concatenate((B1*F1, B2*F1), axis=0)

    C = np.concatenate((H1*C1, H2*C2), axis=1)

    D = np.matrix(H1*D1*F1 + H2*D2*F1)
    
    return signal.StateSpace(A, B, C, D)
