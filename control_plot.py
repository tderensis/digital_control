"""
Plotting functions for control system responses. Pairs best with the output
from control_sim.

Requres matplotlib
"""
import matplotlib.pyplot as plt


def plot_regsf(t, u, x, name=None, plot_now=True):
    """ Plot the response of a full state feedback regulator.
    
    Args:
        t: Time vector
        u: Array of plant inputs
        x: Array of plant state variables
        name (optional): The name to use in the plot_now. Default is None.
        plot_now (optional): Set to True to call plt.show() and False to not.
            Default is True.
            
    Returns:
        Nothing
    """

    (num_samples, num_inputs) = u.shape
    (num_samples, num_states) = x.shape
    
    if name is None:
        name = ""

    plt.figure('Full State Feedback regulator: ' + name)
    plt.subplot(211)
    plt.grid()

    for i in range(0, num_inputs):
        plt.step(t, u[:,i], label='u' + str(i+1))
    plt.legend()
    
    plt.subplot(212)
    plt.grid()

    for i in range(0, num_states):
        plt.step(t, x[:,i], label='x' + str(i+1))
    plt.legend()
    
    if plot_now:
        plt.show()


def plot_regob(t, u, x, xhat, y, name=None, plot_now=True):
    """ Plot the response of a full order observer regulator.
    
    Args:
        t: Time vector
        u: Array of plant inputs
        x: Array of plant state variables
        xhat: Array of estimated state variables
        y: Array of plant outputs
        name (optional): The name to use in the plot_now. Default is None.
        plot_now (optional): Set to True to call plt.show() and False to not.
            Default is True.
    
    Returns:
        Nothing
    """

    (num_samples, num_inputs) = u.shape
    (num_samples, num_states) = x.shape
    (num_samples, num_outputs) = y.shape
    
    if name is None:
        name = ""

    plt.figure('Full order observer regulator: ' + name)
    plt.subplot(411)
    plt.grid()

    for i in range(0, num_inputs):
        plt.step(t, u[:,i], label='u' + str(i+1))
    plt.legend()
    
    plt.subplot(412)
    plt.grid()

    for i in range(0, num_states):
        plt.step(t, x[:,i], label='x' + str(i+1))
    plt.legend()
    
    plt.subplot(413)
    plt.grid()

    for i in range(0, num_states):
        plt.step(t, xhat[:,i], label='xhat' + str(i+1))
    plt.legend()
    
    plt.subplot(414)
    plt.grid()

    for i in range(0, num_outputs):
        plt.step(t, y[:,i], label='y' + str(i+1))
    plt.legend()

    if plot_now:
        plt.show()
