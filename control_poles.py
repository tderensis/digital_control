"""
Tools for choosing poles for closed loop control systems
"""
import cmath


def bessel_spoles(n, Ts=1):
    """ Return the roots of the reverse Bessel polynomial normalized the given
    settling time. The settling time is 1 second by default. Adapted from 
    Digital Control: A State-Space Approach, Table 6.3.
    
    Args:
        n: The order of the Bessel polynomial.
        Ts (optional): The settling time to scale to.

    Returns:
        list: The roots of the Bessel polynomial.
    """

    spoles = [0]
    
    if n == 1:
        spoles = [-4.6200]
    elif n == 2:
        spoles = [-4.0530 + 2.3400j, -4.0530 - 2.3400j]
    elif n == 3:
        spoles = [-5.0093,
                  -3.9668 + 3.7845j, -3.9668 - 3.7845j]
    elif n == 4:
        spoles = [-4.0156 + 5.0723j, -4.0156 - 5.0723j,
                  -5.5281 + 1.6553j, -5.5281 - 1.6553j]
    elif n == 5:
        spoles = [-6.4480,
                  -4.1104 + 6.3142j, -4.1104 - 6.3142j,
                  -5.9268 + 3.0813j, -5.9268 - 3.0813j]
    elif n == 6:
        spoles = [-4.2169 + 7.5300j, -4.2169 - 7.5300j,
                  -6.2613 + 4.4018j, -6.2613 - 4.4018j,
                  -7.1205 + 1.4540j, -7.1205 - 1.4540j]
    elif n == 7:
        spoles = [-8.0271,
                  -4.3361 + 8.7519j, -4.3361 - 8.7519j,
                  -6.5714 + 5.6786j  -6.5714 - 5.6786j,
                  -7.6824 + 2.8081j  -7.6824 - 2.8081j]
    elif n == 8:
        spoles = [-4.4554 + 9.9715j, -4.4554 - 9.9715j,
                  -6.8554 + 6.9278j, -6.8554 - 6.9278j,
                  -8.1682 + 4.1057j, -8.1682 - 4.1057j,
                  -8.7693 + 1.3616j, -8.7693 - 1.3616j]
    elif n == 9:
        spoles = [-9.6585,
                  -4.5696 + 11.1838j, -4.5696 - 11.1838j,
                  -7.1145 +  8.1557j, -7.1145 -  8.1557j,
                  -8.5962 +  5.3655j, -8.5962 -  5.3655j,
                  -9.4013 +  2.6655j, -9.4013 -  2.6655j]
    elif n == 10:
        spoles = [-4.6835 + 12.4022j, -4.6835 - 12.4022j,
                  -7.3609 +  9.3777j, -7.3609 -  9.3777j,
                  -8.9898 +  6.6057j, -8.9898 -  6.6057j,
                  -9.9657 +  3.9342j, -9.9657 -  3.9342j,
                  -10.4278 + 1.3071j, -10.4278 - 1.3071j]

    return [ spole/Ts for spole in spoles ]


def spoles_to_zpoles(spoles, T):
    """ Convert the continouous s-plane poles to the discrete z-plane poles
    with the given sampling interval T.
    
    Args:
        spoles (list): The s-plane poles
        T: The sampling interval in seconds.
    """

    return [ cmath.exp(spole * T) for spole in spoles ]
