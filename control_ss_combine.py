"""
Functions used to combine and connect state space systems. Based on
the paper "Combining and Connection Linear, Multi-Input, Multi-Output
Subsystem Models" by Eugene L. Duke.

Requres numpy
"""
import numpy as np


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
    return None


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
    return None
    
    
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
    return None


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
    return None


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
    return None


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
    return None
