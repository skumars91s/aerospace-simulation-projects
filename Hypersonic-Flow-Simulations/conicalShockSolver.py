import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
import warnings

# Ignore all RuntimeWarnings globally
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# OBLIQUE SHOCK SOLVER (for initial guess)
# ==============================================================================
def solve_beta_weak(theta_deg, M, gamma=1.4):
    if M <= 1: return None
    theta_rad = np.deg2rad(theta_deg)
    def theta_beta_mach_eq(beta_rad):
        numerator = M**2 * np.sin(beta_rad)**2 - 1
        denominator = M**2 * (gamma + np.cos(2 * beta_rad)) + 2
        return np.tan(theta_rad) - (2 * (1 / np.tan(beta_rad)) * numerator / denominator)
    mach_angle_rad = np.arcsin(1 / M)
    initial_guess = mach_angle_rad + np.deg2rad(0.1)
    try:
        beta_rad = fsolve(func=theta_beta_mach_eq, x0=initial_guess)[0]
        beta_deg = np.rad2deg(beta_rad)
        if beta_deg <= np.rad2deg(mach_angle_rad): return None
        return beta_deg
    except (ValueError, IndexError):
        return None

# ==============================================================================
# TAYLOR-MACCOLL SOLVER FUNCTION
# ==============================================================================
def taylor_maccoll(t, y, gamma):
    vr, vt = y[0], y[1]
    if np.abs(np.tan(t)) < 1e-10: cot_t = 1e10
    else: cot_t = 1 / np.tan(t)
    numerator = (-(gamma - 1) / 2 * (1 - vr**2 - vt**2) * (2 * vr + vt * cot_t) + vr * vt**2)
    denominator = ((gamma - 1) / 2 * (1 - vr**2 - vt**2) - vt**2)
    dy = np.array([vt, numerator / denominator])
    return dy
# ==============================================================================
# RK4 FUNCTION
# ==============================================================================
def runge_kutta_rk4(f, t_start, y0, h, gamma):
    t = t_start
    y = y0
    for _ in range(30000): 
        y_prev, t_prev = y, t
        k0 = f(t, y, gamma)
        k1 = f(t + 0.5 * h, y + k0 * 0.5 * h, gamma)
        k2 = f(t + 0.5 * h, y + k1 * 0.5 * h, gamma)
        k3 = f(t + h, y + k2 * h, gamma)
        y = y + (h / 6) * (k0 + 2*k1 + 2*k2 + k3)
        t = t + h
        if np.isnan(y).any(): return None, None, None, None
        if y[1] > 0: return t_prev, y_prev, t, y
    return None, None, None, None

def conicalshocksolver(gamma, Theta_shock, M):
    mach_angle = round(np.rad2deg(np.arcsin(1/M)), 2)
    if Theta_shock <= mach_angle: return 0, None, None
    theta_shock = np.deg2rad(Theta_shock)
    delta = np.arctan(2 * (1/np.tan(theta_shock)) * (((M**2) * (np.sin(theta_shock)**2) - 1) / ((M**2) * (gamma + np.cos(2*theta_shock)) + 2)))
    mn1 = M * np.sin(theta_shock)
    mn2 = np.sqrt(((mn1**2) + (2 / (gamma - 1))) / ((2 * gamma / (gamma - 1)) * (mn1**2) - 1))
    m2 = mn2 / np.sin(theta_shock - delta)
    v_initial = (2 / ((gamma - 1) * (m2**2)) + 1)**(-0.5)
    v_r_initial = v_initial * np.cos(theta_shock - delta)
    v_t_initial = -v_initial * np.sin(theta_shock - delta)
    Y0 = np.array([v_r_initial, v_t_initial])
    h = -0.001
    t_prev, y_prev, t_curr, y_curr = runge_kutta_rk4(taylor_maccoll, theta_shock, Y0, h, gamma)
    if t_prev is None: return 0, None, None
    vt_prev, vt_curr = y_prev[1], y_curr[1]
    t_cone_rad = t_prev + (0 - vt_prev) * (t_curr - t_prev) / (vt_curr - vt_prev)
    vr_prev, vr_curr = y_prev[0], y_curr[0]
    vr_cone = vr_prev + (0 - vt_prev) * (vr_curr - vr_prev) / (vt_curr - vt_prev)
    v_c = vr_cone
    mc = np.sqrt(2 / ((gamma - 1) * (1 / (v_c**2 + 1e-9) - 1)))
    return np.rad2deg(t_cone_rad), m2, mc

# ==============================================================================
# PRESSURE RATIO CALCULATING FUNCTION
# ==============================================================================
def calculate_pressure_ratio(M_inf, shock_angle_deg, M2, Mc, gamma):
    """Calculates Pc/P_inf using the two-step method."""
    beta = np.deg2rad(shock_angle_deg)
    Mn1 = M_inf * np.sin(beta)
    p2_p_inf = 1 + (2 * gamma / (gamma + 1)) * (Mn1**2 - 1)
    term1 = 1 + ((gamma - 1) / 2) * M2**2
    term2 = 1 + ((gamma - 1) / 2) * Mc**2
    pc_p2 = (term1 / term2)**(gamma / (gamma - 1))
    return p2_p_inf * pc_p2

# ==============================================================================
# SHOCK FITTING METHOD SCRIPT
# ==============================================================================
if __name__ == "__main__":
    Mach_array = [2, 4, 6]
    cone_angle_array = np.linspace(5, 40, 70)
    gamma = 1.4
    tolerance, step_size, max_iterations = 1e-4, 0.05, 500
    results = {}

    for M in Mach_array:
        print(f"--- Processing Mach Number M = {M} ---")
        found_results = []
        for target_cone_angle in cone_angle_array:
            shock_guess_deg = solve_beta_weak(target_cone_angle, M)
            if shock_guess_deg is None:
                print(f"   Cone angle {target_cone_angle:.2f}° is too large. Stopping for M={M}.")
                break
            m2_val, mc_val = None, None
            for i in range(max_iterations):
                calculated_cone_angle, m2_val, mc_val = conicalshocksolver(gamma, shock_guess_deg, M)
                if calculated_cone_angle <= 0:
                    shock_guess_deg = None
                    break
                error = calculated_cone_angle - target_cone_angle
                if abs(error) < tolerance:
                    break
                shock_guess_deg -= step_size * np.sign(error)
            
            if shock_guess_deg is not None:
                result_tuple = (shock_guess_deg, m2_val, mc_val)
            else:
                result_tuple = (None, None, None)
            found_results.append(result_tuple)
        results[M] = (cone_angle_array[:len(found_results)], found_results)

    # --- Plotting Section ---

    # ==============================================================================
    # PLOT 1 - SHOCK WAVE ANGLE VS CONE ANGLE ---
    # ==============================================================================
    plt.figure(figsize=(12, 8))
    for M, (cones, res_list) in results.items():
        plot_cones = [c for i, c in enumerate(cones) if res_list[i][0] is not None]
        plot_shocks = [res[0] for res in res_list if res[0] is not None]
        plt.plot(plot_cones, plot_shocks, label=f'M = {M}')
    plt.title('Conical Shock Wave Angle vs. Cone Angle', fontsize=16)
    plt.xlabel('Cone Semi-Vertex Angle, $θ_c$ (degrees)', fontsize=12)
    plt.ylabel('Shock Wave Angle, $β$ (degrees)', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend(fontsize=12)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # ==============================================================================
    # PLOT 2 - PRESSURE RATIO VS CONE ANGLE ---
    # ==============================================================================
    plt.figure(figsize=(12, 8))
    for M, (cones, res_list) in results.items():
        # Prepare lists for plotting
        plot_cones = []
        plot_pressure_ratios = []
        
        for i, res in enumerate(res_list):
            shock_angle, m2, mc = res
            if shock_angle is not None:
                # Get the corresponding cone angle
                cone_angle = cones[i]
                
                # Calculate the pressure ratio
                pc_p_inf = calculate_pressure_ratio(M, shock_angle, m2, mc, gamma)
                
                # Add the data to our lists
                plot_cones.append(cone_angle)
                plot_pressure_ratios.append(pc_p_inf)
                
        # Plot the data for the current Mach number
        plt.plot(plot_cones, plot_pressure_ratios, label=f'M = {M}')

    # --- Finalize and show the plot ---
    plt.title('Cone Surface Pressure Ratio vs. Cone Angle', fontsize=16)
    plt.xlabel('Cone Semi-Vertex Angle, $θ_c$ (degrees)', fontsize=12)
    plt.ylabel('Surface Pressure Ratio, $P_c / P_{\infty}$', fontsize=12)
    plt.grid(True, which='both', linestyle='--')
    plt.legend(fontsize=12)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Show both plots
    plt.show()