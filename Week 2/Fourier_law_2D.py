import torch
import matplotlib.pyplot as plt

"""
 2D heat conduction equation (Fourier's law)

 The heat equation is a partial differential equation (PDE) that describes the distribution of heat (or temperature) in a given region over time.
"""

# Define the domain and parameters
Lx = 10  # Length of the plate in x-direction (meters)
Ly = 10  # Length of the plate in y-direction (meters)
Nx = 50  # Number of points in x-direction
Ny = 50  # Number of points in y-direction
dx = Lx / (Nx - 1)  # Spatial step in x-direction
dy = Ly / (Ny - 1)  # Spatial step in y-direction
alpha = 0.01  # Thermal diffusivity (m^2/s)
dt = 0.1  # Time step (seconds)
T_max = 20  # Maximum simulation time (seconds)

# Initialize spatial grid
x = torch.linspace(0, Lx, Nx)
y = torch.linspace(0, Ly, Ny)
X, Y = torch.meshgrid(x, y)

# Initial temperature distribution (e.g., a Gaussian pulse in the center)
u_initial = torch.exp(-((X - Lx / 2)**2 + (Y - Ly / 2)**2) / 0.1)  # Gaussian

# Convert to PyTorch tensor
u = u_initial.clone().detach().requires_grad_(False)  # Temperature at each spatial point
u_new = u.clone()

# Precompute constant for the finite difference method
rx = alpha * dt / dx**2
ry = alpha * dt / dy**2

# Function to update the temperature using finite differences (2D)
def heat_transfer_step_2d(u, rx, ry):
    u_new = u.clone()
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            u_new[i, j] = u[i, j] + rx * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) + \
                            ry * (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])
    return u_new

# Time-stepping loop
timesteps = int(T_max / dt)
u_all = [u_initial]

# Run the simulation and store results for visualization
for t in range(timesteps):
    u = heat_transfer_step_2d(u, rx, ry)  # Update the temperature distribution
    if t % 100 == 0:  # Store every 100 steps for visualization
        u_all.append(u.clone())

# Plot the temperature distribution at different time steps
plt.figure(figsize=(12, 8))
for i, u_t in enumerate(u_all):
    plt.subplot(3, 1, i+1)
    plt.contourf(X.numpy(), Y.numpy(), u_t.numpy(), 20, cmap='inferno')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Time = {i*100*dt:.2f} s')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)

plt.tight_layout()
plt.show(block=True)
