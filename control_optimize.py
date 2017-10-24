"""
Functions that attempt to maximize the stability margins of the
control system while keeping the same settling time.

Requires scipy
"""
from scipy import signal
from scipy.optimize import minimize, brute, fmin
import control_sim, control_design, control_eval
import cmath

def optimize_regsf(phi, gamma, T, Ts):
    
    # Function to optimize
    def stability_measure(input):
        zpoles = [complex(input[0], 0), complex(input[1], input[2]), complex(input[1], -input[2])]
        for zpole in zpoles:
            if abs(zpole) > 1:
                return 100
        # TODO: for now we check for duplicates and don't allow them. When FBG is done,
        # then use that
        if len(zpoles) != len(set(zpoles)):
            return 100

        fsf = signal.place_poles(phi, gamma, zpoles)
        L = fsf.gain_matrix
        ugms = control_eval.upper_gain_margin(phi, gamma, L, output_dB=False)
        lgms = control_eval.lower_gain_margin(phi, gamma, L, output_dB=False)
        phms = control_eval.phase_margin(phi, gamma, L)
        (t, u, x) = control_sim.sim_regsf(phi, gamma, L, T, x0, Ts*3)
        st = control_eval.settling_time(t, x)
        st_diff = abs(st-Ts)
        gm_max = 1000
        phm_max = 120
        st_max = Ts
        ugm_percent = 10/gm_max
        lgm_percent = 0/gm_max
        phm_percent= 1/phm_max
        st_percent = 4/Ts
        result = 0
        for i in range(0, len(ugms)):
            result += lgms[i]*lgm_percent - ugms[i]*ugm_percent - phms[0]*phm_percent + st_diff*st_percent

        return result
    
    # Start with bessel poles
    spoles = control_design.bessel_spoles(phi.shape[0], Ts)
    zpoles = control_design.spoles_to_zpoles(spoles, T)
    x0 = [ zpoles[0].real, zpoles[1].real, zpoles[1].imag ]
    
    min_methods = [ 'nelder-mead' ] #, 'brute' ]
    best_result = 0
    for method in min_methods:
        if method == 'brute':
            resbrute = brute(stability_measure, ((x0[0]-0.2,1),(x0[1]-0.2,1),(0.01,x0[2]+0.2)), Ns=20, full_output=True, finish=None)
            #x = resbrute[0]
            #zpoles = [complex(x[0],0), complex(x[1], x[2]), complex(x[1], -x[2])]
            #function_value = resbrute[1]
            res = minimize(stability_measure, resbrute[0], method='nelder-mead', tol=1e-8)
            function_value = res.fun
            zpoles = [complex(res.x[0],0), complex(res.x[1], res.x[2]), complex(res.x[1], -res.x[2])]
        else:
            res = minimize(stability_measure, x0, method=method, tol=1e-8)
            function_value = res.fun
            zpoles = [complex(res.x[0],0), complex(res.x[1], res.x[2]), complex(res.x[1], -res.x[2])]

        print(function_value)
        print(zpoles)
        if best_result > function_value:
            best_result = function_value
            fsf = signal.place_poles(phi, gamma, zpoles)
            L = fsf.gain_matrix

    return L
    