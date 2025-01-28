import torch
import matplotlib.pyplot as plt
import numpy as np

# Parameters
g = 9.81  # Gravitational acceleration (m/s^2)
l = 1.0   # Length of pendulum (m)
b = 1.1   # Damping coefficient
dt = 0.01 # Time step

# Dynamics function
def pendulum_dynamics(state):
    theta, omega = state
    dtheta = omega
    domega = -b * omega - (g / l) * torch.sin(theta)
    return torch.tensor([dtheta, domega])

# Simulate the pendulum
def simulate_pendulum(theta0, omega0, steps):
    state = torch.tensor([theta0, omega0], dtype=torch.float32)
    trajectory = [state.clone()]
    for _ in range(steps):
        state += pendulum_dynamics(state) * dt
        trajectory.append(state.clone())
    return torch.stack(trajectory)

# Vector field for the pendulum
def generate_vector_field(grid_size=20):
    theta_range = torch.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    omega_range = torch.linspace(-10, 10, grid_size)
    theta, omega = torch.meshgrid(theta_range, omega_range)
    theta, omega = theta.flatten(), omega.flatten()
    field = torch.stack([theta, omega], dim=1)
    vectors = torch.stack([pendulum_dynamics(f) for f in field])
    return theta, omega, vectors[:, 0], vectors[:, 1]

# Visualization
def plot_results(trajectory, vector_field):
    theta = trajectory[:, 0].numpy()
    omega = trajectory[:, 1].numpy()
    time = np.arange(len(theta)) * dt

    # Time series
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(time, theta, label="Theta (Angle)")
    plt.plot(time, omega, label="Omega (Angular Velocity)")
    plt.title("Time Series")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.legend()
    plt.grid()

    # Phase diagram
    plt.subplot(1, 3, 2)
    plt.plot(theta, omega, label="Trajectory")
    plt.title("Phase Diagram")
    plt.xlabel("Theta (Angle)")
    plt.ylabel("Omega (Angular Velocity)")
    plt.grid()

    # Vector field
    plt.subplot(1, 3, 3)
    theta_grid, omega_grid, dtheta, domega = vector_field
    plt.quiver(
        theta_grid.numpy(), omega_grid.numpy(),
        dtheta.numpy(), domega.numpy(), color="blue", alpha=0.7
    )
    plt.title("Vector Field")
    plt.xlabel("Theta (Angle)")
    plt.ylabel("Omega (Angular Velocity)")
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.show(block=True)

# Main
if __name__ == "__main__":
    # Initial conditions
    theta0 = 1.0  # Initial angle (radians)
    omega0 = 0.0  # Initial angular velocity (rad/s)
    steps = 1000

    # Simulate and visualize
    trajectory = simulate_pendulum(theta0, omega0, steps)
    vector_field = generate_vector_field(grid_size=20)
    plot_results(trajectory, vector_field)
