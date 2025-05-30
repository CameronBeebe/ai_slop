import math

import time

import os


# --- Lotka-Volterra Model Parameters ---

# Prey (x)

alpha = 1.5  # Intrinsic growth rate of prey

beta = 0.1   # Predation rate (how effectively predators feed on prey)


# Predator (y)

gamma = 0.8  # Death rate of predators

delta = 0.05 # Prey consumption conversion efficiency (how efficiently prey become new predators)


# --- Initial Conditions ---

x0 = 50.0  # Initial prey population

y0 = 10.0  # Initial predator population


# --- Simulation Parameters ---

dt = 0.01      # Time step

t_max = 50.0   # Total simulation time

num_steps = int(t_max / dt)


# --- Visualization Parameters ---

viz_width = 60  # Width of the ASCII visualization

viz_height_max = 100 # Maximum population value for visualization scaling


# --- Simulation Function ---

def lotka_volterra(x, y):

    """

    Calculates the rate of change for prey (x) and predator (y) populations

    based on the Lotka-Volterra equations.

    """

    dx_dt = alpha * x - beta * x * y

    dy_dt = delta * x * y - gamma * y

    return dx_dt, dy_dt


# --- Visualization Function ---

def clear_screen():

    """Clears the terminal screen."""

    os.system('cls' if os.name == 'nt' else 'clear')


def visualize_populations(x, y, t):

    """

    Generates a simple ASCII visualization of the populations.

    """

    clear_screen()

    print(f"--- Lotka-Volterra Simulation ---")

    print(f"Time: {t:.2f}")

    print(f"Prey (x): {x:.2f}")

    print(f"Predators (y): {y:.2f}")

    print("-" * viz_width)


    # Scale populations for visualization

    max_pop = max(x, y)

    if max_pop == 0:

        scale_factor = 0

    else:

        scale_factor = viz_height_max / max_pop


    x_viz = int(x * scale_factor)

    y_viz = int(y * scale_factor)


    # Create the visualization bars

    x_bar = '#' * min(x_viz, viz_width - 2)

    y_bar = '*' * min(y_viz, viz_width - 2)


    # Ensure bars are centered and don't exceed the width

    x_bar = x_bar.ljust(viz_width - 2)

    y_bar = y_bar.ljust(viz_width - 2)


    # Print the visualization

    print(f"Prey:       [{x_bar}]")

    print(f"Predators:  [{y_bar}]")

    print("-" * viz_width)

    print("Legend: # = Prey, * = Predators")


# --- Simulation Loop ---

x = x0

y = y0

t = 0.0


print("Starting Lotka-Volterra Simulation...")

print(f"Initial conditions: x={x0:.2f}, y={y0:.2f}")

print(f"Parameters: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")

print(f"Simulation steps: dt={dt}, t_max={t_max}")

time.sleep(2) # Pause for a moment before starting


for i in range(num_steps):

    # Calculate derivatives using the Lotka-Volterra equations

    dx_dt, dy_dt = lotka_volterra(x, y)


    # Update populations using the Euler method (simple approximation)

    # x_new = x_old + dt * f(x_old, y_old)

    x_new = x + dt * dx_dt

    y_new = y + dt * dy_dt


    # Ensure populations don't become negative

    x = max(0.0, x_new)

    y = max(0.0, y_new)


    # Update time

    t += dt


    # Visualize every 10 steps (or at the last step)

    if i % 10 == 0 or i == num_steps - 1:

        visualize_populations(x, y, t)

        time.sleep(0.05) # Small delay to make visualization visible


print("\nSimulation finished.")

print(f"Final time: t={t:.2f}")

print(f"Final populations: x={x:.2f}, y={y:.2f}")
