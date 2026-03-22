import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
from Rotor import Rotor
from globals import data_dir

def normalize(x, bounds):
    """Normalize parameters to [0,1] range"""
    x_norm = [(x[i]-bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x))]
    bounds_norm = [[0,1] for i in range(len(x))]
    return x_norm, bounds_norm

def denormalize(x_norm, bounds):
    """Denormalize parameters back to original range"""
    x = [
        bounds[i][0] + x_norm[i] * (bounds[i][1] - bounds[i][0])
        for i in range(len(x_norm))
    ]
    return x

def objective_energy_harvesting(x_norm, args, debug=False):
    """
    Objective function for energy harvesting optimization.
    Returns power (should be negative for harvesting).
    We minimize this to maximize energy extraction.
    """
    B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds = args
    x = denormalize(x_norm, bounds)
    at, bt, ac, bc = x
    
    twst_func = lambda r_R: at*r_R + bt
    c_R_func = lambda r_R: bc + ac*r_R
    
    try:
        # Create and evaluate rotor (updated for new Rotor class)
        rotor = Rotor(
            c_R_func=c_R_func,
            twst_func=twst_func,
            pitch=0,
            B=B,
            R=R,
            r_R_H=r_R_H,
            polar_path=polar_path,
            J=J,
            Vinf=Uinf,
            rho=rho,
            n_elem=n_elem,
            dist_elem=dist_elem,
            isPropeller=isPropeller
        )
        
        # Get power directly from rotor attribute (no calculate_integral)
        power = rotor.P
        
        # Extract aerodynamic data for penalties and monitoring
        alphas = rotor.alpha
        a_values = rotor.a
        
        # Calculate penalties and incentives
        penalty = 0
        incentive = 0
        
        # 1. Physics check: Theoretical maximum power (Betz limit)
        area = np.pi * R**2
        P_betz = 0.5 * rho * Uinf**3 * area * (16/27)  # ~59.5 kW for your case
        
        if abs(power) > P_betz * 1.2:  # Allow 20% margin for numerical effects
            penalty += 1e6 * (abs(power) / P_betz)
            if debug:
                print(f"  WARNING: Unphysical power: {power:.0f} W (Betz: {P_betz:.0f} W)")
        
        # 2. Strong penalty for positive power (should be harvesting)
        if power > 0:
            penalty += 100000 * (power / 1000)
            if debug:
                print(f"  WARNING: Positive power: {power:.0f} W")
        
        # 3. Encourage negative induction (windmill state)
        if len(a_values) > 0:
            avg_a = np.mean(a_values)
            if avg_a >= 0:
                penalty += 50000 * (avg_a + 0.1)
            else:
                incentive += 1000 * abs(avg_a)  # Reward negative induction
        
        # 4. Encourage alpha in optimal range for harvesting (-5 to -10 degrees)
        if len(alphas) > 0:
            avg_alpha = np.mean(alphas)
            # Target alpha for maximum power extraction in windmill state
            target_alpha = -7  # degrees
            
            if avg_alpha > -3:
                penalty += 5000 * (avg_alpha + 3)
            elif avg_alpha < -12:
                penalty += 2000 * (-12 - avg_alpha)
            else:
                # Reward being in optimal range
                incentive += 1000 * (1 - abs(avg_alpha - target_alpha) / 10)
        
        # 5. Penalize unrealistic chord distributions
        chord_tip = bc + ac * 1.0
        chord_root = bc + ac * 0.25
        
        if chord_tip < 0.03 or chord_root < 0.05:
            penalty += 100000
        
        if chord_tip > chord_root:
            penalty += 10000  # Penalize reverse taper for harvesting
        
        # 6. Penalize twist that leads to stall or poor performance
        twist_tip = at * 1.0 + bt
        twist_root = at * 0.25 + bt
        
        if twist_tip < -20 or twist_tip > 40:
            penalty += 10000
        if twist_root < 10 or twist_root > 120:
            penalty += 10000
        
        # Scale power to kW for better numerical behavior
        scaled_power = power / 1000.0
        
        # Combined objective (minimize)
        objective_value = scaled_power + penalty / 1000.0 - incentive / 1000.0
        
        # Optional debug output
        if debug and np.random.random() < 0.1:  # 10% of evaluations
            Cp = power / (0.5 * rho * Uinf**3 * area)
            print(f"  [DEBUG] at={at:.1f}, bt={bt:.1f}, ac={ac:.4f}, bc={bc:.4f}")
            print(f"          Power={power:.0f}W, Cp={Cp:.3f}, a_mean={np.mean(a_values):.3f}")
            print(f"          alpha_mean={np.mean(alphas):.1f}°, obj={objective_value:.2f}")
        
        return objective_value
        
    except Exception as e:
        if debug:
            print(f"  ERROR in objective: {e}")
        return 1e6  # Large penalty for failed designs

def optimize_energy_harvesting():
    """
    Main optimization routine for energy harvesting blade design
    """
    print("="*80)
    print("ENERGY HARVESTING OPTIMIZATION")
    print("Optimizing twist and chord distributions for maximum power extraction")
    print("="*80)
    
    # Define design variables bounds
    bounds = [
        [-90, 90],    # at - twist slope (deg per r/R)
        [10, 120],    # bt - twist intercept (deg at root)
        [-0.10, -0.02],  # ac - chord slope (negative = taper)
        [0.10, 0.30]     # bc - chord intercept (c/R at root)
    ]
    
    # Initial design
    x0 = [-50, 81, -0.06, 0.18]
    
    # Normalize initial guess
    x0_norm, bounds_norm = normalize(x0, bounds)
    
    # Operating conditions
    B = 6                    # Number of blades
    R = 0.7                  # Radius (m)
    r_R_H = 0.25             # Hub radius ratio
    n_elem = 100             # Number of blade elements
    polar_path = data_dir.joinpath('ARAD8pct_polar.txt')
    dist_elem = "uniform"
    isPropeller = True       # Propeller mode (for energy harvesting)
    Uinf = 60                # Freestream velocity (m/s) - landing speed
    rho = 1.067              # Air density (kg/m³)
    J = 60/20/1.4           # Advance ratio
    
    arguments = [B, J, r_R_H, polar_path, n_elem, dist_elem, isPropeller, R, Uinf, rho, bounds]
    
    # Test initial design
    print("\n" + "-"*80)
    print("EVALUATING INITIAL DESIGN")
    print("-"*80)
    
    initial_power = objective_energy_harvesting(x0_norm, arguments, debug=True)
    initial_power_watts = initial_power * 1000  # Convert back from kW
    
    print(f"\nInitial design performance:")
    print(f"  Power: {initial_power_watts:.2f} W")
    if initial_power_watts < 0:
        print(f"  Status: ✓ Energy harvesting ({abs(initial_power_watts):.2f} W extracted)")
    else:
        print(f"  Status: ✗ Power consumption ({initial_power_watts:.2f} W consumed)")
    
    # Strategy 1: Differential Evolution for global search
    print("\n" + "="*80)
    print("STAGE 1: DIFFERENTIAL EVOLUTION (Global Search)")
    print("="*80)
    print("Exploring the design space to find promising regions...")
    
    try:
        res_de = differential_evolution(
            func=objective_energy_harvesting,
            args=arguments,
            bounds=bounds_norm,
            strategy='rand1bin',      # More exploratory strategy
            maxiter=80,               # Number of generations
            popsize=12,               # Population size
            tol=0.01,                 # Tolerance for convergence
            mutation=(0.5, 1.5),      # Mutation range
            recombination=0.7,        # Crossover probability
            seed=42,                  # For reproducibility
            workers=1,                # Disable parallelization
            disp=True,                # Display progress
            polish=False              # Don't polish yet
        )
        
        print(f"\nDifferential Evolution results:")
        print(f"  Success: {res_de.success}")
        print(f"  Best objective value: {res_de.fun:.2f}")
        print(f"  Number of evaluations: {res_de.nfev}")
        
        # Denormalize DE result for inspection
        x_de = denormalize(res_de.x, bounds)
        print(f"  Best parameters: at={x_de[0]:.1f}, bt={x_de[1]:.1f}, ac={x_de[2]:.4f}, bc={x_de[3]:.4f}")
        
        # Calculate actual power for DE result
        de_power_obj = objective_energy_harvesting(res_de.x, arguments)
        de_power_watts = de_power_obj * 1000
        print(f"  Power: {de_power_watts:.2f} W")
        
        best_x_norm = res_de.x
        best_obj = res_de.fun
        
    except Exception as e:
        print(f"  Differential Evolution failed: {e}")
        print("  Falling back to SLSQP from initial point...")
        best_x_norm = x0_norm
        best_obj = initial_power
    
    # Strategy 2: Local refinement with SLSQP
    print("\n" + "="*80)
    print("STAGE 2: SLSQP LOCAL REFINEMENT")
    print("="*80)
    print("Refining the best solution with gradient-based optimization...")
    
    # Try multiple initial points for SLSQP
    initial_points = [best_x_norm]
    
    # Add perturbed versions of the best point
    for i in range(3):
        perturbation = np.random.normal(0, 0.1, len(best_x_norm))
        perturbed = np.clip(best_x_norm + perturbation, 0, 1)
        initial_points.append(perturbed)
    
    best_slsqp_result = None
    best_slsqp_obj = float('inf')
    
    for i, start_point in enumerate(initial_points):
        print(f"\n  SLSQP attempt {i+1}/{len(initial_points)}...")
        
        res = minimize(
            fun=objective_energy_harvesting,
            args=arguments,
            x0=start_point,
            method="SLSQP",
            bounds=bounds_norm,
            options={
                'maxiter': 500,
                'ftol': 1e-8,
                'eps': 0.05,           # Larger step for gradient
                'disp': False          # Don't display individual results
            }
        )
        
        if res.success and res.fun < best_slsqp_obj:
            best_slsqp_obj = res.fun
            best_slsqp_result = res
            print(f"    ✓ New best found: objective={res.fun:.2f}")
    
    # Use the best SLSQP result if available
    if best_slsqp_result is not None and best_slsqp_result.fun < best_obj:
        final_result = best_slsqp_result
        print(f"\n  SLSQP improved the solution from {best_obj:.2f} to {final_result.fun:.2f}")
    else:
        final_result = type('obj', (), {'x': best_x_norm, 'fun': best_obj, 'success': True, 'message': 'Using DE result'})()
        print(f"\n  SLSQP did not improve beyond DE result")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL OPTIMIZATION RESULTS")
    print("="*80)
    
    # Denormalize final parameters
    x_norm = final_result.x
    x = denormalize(x_norm, bounds)
    at, bt, ac, bc = x
    
    print(f"\nOptimized blade parameters:")
    print(f"  Twist distribution: θ(r/R) = {at:.2f}·(r/R) + {bt:.2f}°")
    print(f"  Chord distribution: c(r/R)/R = {bc:.4f} + {ac:.4f}·(r/R)")
    print(f"  Tip chord: {(bc + ac)*R*1000:.1f} mm")
    print(f"  Root chord: {(bc + ac*0.25)*R*1000:.1f} mm")
    print(f"  Taper ratio: {(bc + ac)/(bc + ac*0.25):.3f}")
    
    # Create final rotor for detailed analysis
    twst_func = lambda r_R: at*r_R + bt
    c_R_func = lambda r_R: bc + ac*r_R
    
    rotor = Rotor(
        c_R_func=c_R_func,
        twst_func=twst_func,
        pitch=0,
        B=B,
        R=R,
        r_R_H=r_R_H,
        polar_path=data_dir.joinpath('ARAD8pct_polar.txt'),
        J=J,
        Vinf=Uinf,
        rho=rho,
        n_elem=n_elem,
        dist_elem='uniform',
        isPropeller=isPropeller
    )
    
    # Calculate final performance (access attributes directly)
    final_power = rotor.P
    thrust = rotor.T
    torque = rotor.Q
    
    print(f"\nFinal aerodynamic performance:")
    print(f"  Power extracted: {final_power:.2f} W")
    
    if final_power < 0:
        print(f"  Status: ✓ Energy harvesting ({abs(final_power):.2f} W extracted)")
    else:
        print(f"  Status: ✗ Power consumption ({final_power:.2f} W consumed)")
    
    print(f"  Thrust: {thrust:.1f} N")
    print(f"  Torque: {torque:.1f} N·m")
    
    # Calculate power coefficient
    area = np.pi * R**2
    Cp = final_power / (0.5 * rho * Uinf**3 * area)
    Cp_betz = 16/27  # 0.5926
    
    print(f"  Power coefficient (Cp): {Cp:.4f}")
    print(f"  Betz limit: {Cp_betz:.4f}")
    print(f"  Efficiency relative to Betz: {(Cp/Cp_betz)*100:.1f}%")
    
    # Extract and display aerodynamic data
    alphas = rotor.alpha
    a_values = rotor.a
    
    if len(alphas) > 0:
        print(f"\nAngle of attack distribution:")
        print(f"  Range: {np.min(alphas):.2f}° to {np.max(alphas):.2f}°")
        print(f"  Average: {np.mean(alphas):.2f}°")
        print(f"  At root (r/R=0.25): {alphas[0]:.2f}°")
        print(f"  At tip (r/R=1.0): {alphas[-1]:.2f}°")
        
        print(f"\nInduction factors:")
        print(f"  Axial induction (a) range: {np.min(a_values):.4f} to {np.max(a_values):.4f}")
        print(f"  Average a: {np.mean(a_values):.4f}")
        
        # Check windmill state
        if np.mean(a_values) < 0:
            print(f"  ✓ Windmill state confirmed (negative induction)")
        else:
            print(f"  ⚠ Not in windmill state (positive induction)")
        
        # Check alpha constraints
        alpha_min = -15
        alpha_max = 20
        if np.any(alphas < alpha_min):
            print(f"  ⚠ Warning: {np.sum(alphas < alpha_min)} elements below {alpha_min}°")
        if np.any(alphas > alpha_max):
            print(f"  ⚠ Warning: {np.sum(alphas > alpha_max)} elements above {alpha_max}°")
        else:
            print(f"  ✓ All alphas within [{alpha_min}°, {alpha_max}°]")
    
    # Compare with initial design
    print(f"\nImprovement summary:")
    print(f"  Initial power: {initial_power_watts:.2f} W")
    print(f"  Final power: {final_power:.2f} W")
    
    if final_power < initial_power_watts:
        improvement = ((initial_power_watts - final_power) / abs(initial_power_watts)) * 100
        print(f"  Improvement: {improvement:.1f}% more energy harvested")
    else:
        print(f"  No improvement achieved")
    
    # Plot results
    print("\nGenerating plots...")
    rotor.plot_check()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    return final_result, rotor

# Run the optimization
if __name__ == "__main__":
    result, optimized_rotor = optimize_energy_harvesting()