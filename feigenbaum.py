import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Feigenbaum Constants (Approximate) ---
FEIGENBAUM_DELTA = 4.669201609
FEIGENBAUM_ALPHA  = 2.564949570

def logistic_map(x, r):
    """Performs one iteration of the logistic map."""
    return r * x * (1 - x)

def generate_sequence(r, x0, num_iterations, transient_iterations):
    """
    Generates a sequence of logistic map values for a given r and x0.
    Skips the first 'transient_iterations' to reach stable behavior.
    """
    x = x0
    for _ in range(transient_iterations):
        x = logistic_map(x, r)

    sequence = []
    for _ in range(num_iterations):
        x = logistic_map(x, r)
        sequence.append(x)
    return sequence

def bifurcation_diagram(r_min, r_max, num_r, x0, num_iterations, transient_iterations):
    """
    Generates a bifurcation diagram for the logistic map.
    """
    r_values = np.linspace(r_min, r_max, num_r)
    x_values = []
    r_all = []

    print(f"Generating bifurcation diagram for {num_r} r values...")
    for i, r in enumerate(r_values):
        sequence = generate_sequence(r, x0, num_iterations, transient_iterations)
        x_values.extend(sequence)
        r_all.extend([r] * len(sequence))

        if (i + 1) % (num_r // 10) == 0:
            print(f"  Progress: {i + 1}/{num_r} completed.")

    print("Generation complete.")
    return np.array(r_all), np.array(x_values)

def plot_bifurcation_diagram(r_values, x_values):
    """
    Plots the bifurcation diagram.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(r_values, x_values, ',', alpha=0.3, markersize=0.5)
    plt.xlabel("Growth Rate (r)")
    plt.ylabel("Population Fraction (x)")
    plt.title("Logistic Map Bifurcation Diagram")
    plt.ylim(0, 1)
    plt.xlim(0, 4)
    plt.grid(True, alpha=0.3)
    plt.show()

def estimate_feigenbaum_constants(r_min, r_max, num_r, x0, num_iterations, transient_iterations):
    """
    Estimates Feigenbaum constants by finding successive period-doubling bifurcation points.
    """
    r_values = np.linspace(r_min, r_max, num_r)
    
    # We need to find the stable points (or groups of points) for each r
    # and look for the differences in the number of points as r increases.
    # A simple approach is to look at the peaks in the bifurcation diagram.
    
    # Let's focus on the region where period-doubling occurs.
    # The first period-doubling bifurcation starts around r ~ 3.0
    
    r_points = []
    
    # Find points where the number of stable points doubles
    # This is a heuristic approach; more robust methods might involve finding fixed points.
    
    # Let's do a coarse scan first to find approximate bifurcation points
    # and then refine the search around them.
    
    last_num_points = 0
    
    for i, r in enumerate(r_values):
        sequence = generate_sequence(r, x0, num_iterations, transient_iterations)
        
        # Count unique points in the final sequence
        unique_points = np.unique(np.round(sequence, decimals=4)) # Round to group similar points
        num_points = len(unique_points)
        
        # If the number of points doubles (or more), record the r
        if last_num_points > 0 and num_points > last_num_points * 1.5:
            r_points.append(r)
            last_num_points = num_points
        
        # Reset if the number of points drops significantly (e.g., entering chaos)
        elif num_points < last_num_points * 0.5 and last_num_points > 1:
             last_num_points = num_points
        
        # Initial condition
        elif last_num_points == 0 and num_points > 1:
             last_num_points = num_points
             
        # Handle cases where the number of points becomes 1 after chaos
        elif last_num_points > 1 and num_points == 1:
             last_num_points = 1 # Reset to look for next doubling
        
        # Print progress
        if (i + 1) % (num_r // 10) == 0 and len(r_points) > 1:
            print(f"  Estimating constants: {i + 1}/{num_r} completed. Found {len(r_points)} bifurcation points so far.")

    # Refine the bifurcation points by looking at the peaks in the diagram
    # or finding points where the number of points drops to 1 after chaos
    # or grows rapidly.
    
    # Let's use a simpler approach: find the r values where the number of points
    # roughly transitions from 1 to 2, 2 to 4, 4 to 8, etc.
    
    r_bifurcations = []
    
    # Find the first bifurcation point (1 -> 2)
    sequence_0 = generate_sequence(r_values[0], x0, num_iterations, transient_iterations)
    unique_points_0 = np.unique(np.round(sequence_0, decimals=4))
    last_num_points = len(unique_points_0)
    
    for i in range(1, num_r):
        r = r_values[i]
        sequence = generate_sequence(r, x0, num_iterations, transient_iterations)
        unique_points = np.unique(np.round(sequence, decimals=4))
        num_points = len(unique_points)
        
        # Check if it looks like a doubling point
        if last_num_points > 0 and num_points > last_num_points * 1.5:
            r_bifurcations.append(r)
            last_num_points = num_points
        
        # Reset if the number of points drops significantly (entering chaos)
        elif num_points < last_num_points * 0.5 and last_num_points > 1:
             last_num_points = num_points
        
        # Handle cases where the number of points becomes 1 after chaos
        elif last_num_points > 1 and num_points == 1:
             last_num_points = 1 # Reset to look for next doubling
        
        # Initial condition
        elif last_num_points == 0 and num_points > 1:
             last_num_points = num_points
             
    if len(r_bifurcations) < 2:
        print("Warning: Not enough bifurcation points found to estimate constants.")
        return None, None

    print(f"Found {len(r_bifurcations)} potential bifurcation points.")
    print(f"Bifurcation points: {r_bifurcations}")
    
    # Calculate differences between successive bifurcation points
    r_diffs = np.diff(r_bifurcations)
    
    if len(r_diffs) < 2:
        print("Warning: Not enough differences found to estimate constants.")
        return None, None

    # Calculate ratios of successive differences
    delta_ratios = r_diffs[1:] / r_diffs[:-1]
    
    # Estimate delta as the average of the ratios
    estimated_delta = np.mean(delta_ratios)
    
    # Estimate alpha using the first two differences
    if len(r_diffs) >= 2:
        estimated_alpha = r_diffs[1] / r_diffs[0]
    else:
        estimated_alpha = None

    return estimated_delta, estimated_alpha

# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    r_min = 0.0
    r_max = 4.0
    num_r = 10000  # Number of r values to test
    x0 = 0.5       # Initial population fraction
    num_iterations = 1000 # Number of iterations for each r
    transient_iterations = 200 # Iterations to discard at the start

    # --- Command Line Arguments (Optional) ---
    if len(sys.argv) > 1:
        try:
            r_min = float(sys.argv[1])
            r_max = float(sys.argv[2])
            num_r = int(sys.argv[3])
            x0 = float(sys.argv[4])
            num_iterations = int(sys.argv[5])
            transient_iterations = int(sys.argv[6])
        except (ValueError, IndexError):
            print("Usage: python feigenbaum.py [r_min] [r_max] [num_r] [x0] [num_iterations] [transient_iterations]")
            print("Using default parameters.")

    # --- Generate Bifurcation Diagram ---
    r_values, x_values = bifurcation_diagram(
        r_min, r_max, num_r, x0, num_iterations, transient_iterations
    )

    # --- Plot Bifurcation Diagram ---
    plot_bifurcation_diagram(r_values, x_values)

    # --- Estimate Feigenbaum Constants ---
    print("\nEstimating Feigenbaum constants...")
    # Focus on the region where period-doubling is clearly visible
    r_est_min = 2.5
    r_est_max = 3.8
    num_est_r = 20000 # More points for better estimation
    num_est_iterations = 2000
    transient_est_iterations = 500

    estimated_delta, estimated_alpha = estimate_feigenbaum_constants(
        r_est_min, r_est_max, num_est_r, x0, num_est_iterations, transient_est_iterations
    )

    if estimated_delta is not None and estimated_alpha is not None:
        print(f"\n--- Estimated Feigenbaum Constants ---")
        print(f"Estimated Delta (δ): {estimated_delta:.8f}")
        print(f"Actual Delta (δ):    {FEIGENBAUM_DELTA:.8f}")
        print(f"Difference:          {abs(estimated_delta - FEIGENBAUM_DELTA):.8e}")
        print(f"\nEstimated Alpha (α): {estimated_alpha:.8f}")
        print(f"Actual Alpha (α):    {FEIGENBAUM_ALPHA:.8f}")
        print(f"Difference:          {abs(estimated_alpha - FEIGENBAUM_ALPHA):.8e}")
    else:
        print("\nCould not reliably estimate Feigenbaum constants.")

    # --- Connection to Fractals ---
    print("\n--- Connection to Fractals ---")
    print("The bifurcation diagram, especially in the chaotic region (r > ~3.57),")
    print("exhibits fractal properties. If you zoom in on certain parts of the diagram,")
    print("you often see smaller copies of the overall structure, including period-doubling")
    print("and chaotic regions. This self-similarity is a hallmark of fractals.")
    print("The Feigenbaum constants arise from the scaling properties of these fractal structures.")
