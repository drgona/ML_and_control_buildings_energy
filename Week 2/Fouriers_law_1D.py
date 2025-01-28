import torch
import matplotlib.pyplot as plt



"""
 1D heat conduction equation (Fourier's law)
 
 The heat equation is a partial differential equation (PDE) that describes the distribution of heat (or temperature) in a given region over time.
"""


# Define the domain and parameters
L = 10  # Length of the rod (meters)
Nx = 100  # Number of spatial points (discretization points)
dx = L / (Nx - 1)  # Spatial step size
alpha = 0.01  # Thermal diffusivity (m^2/s)
dt = 0.1  # Time step (seconds)
T_max = 100  # Maximum simulation time (seconds)

# Initialize spatial grid
x = torch.linspace(0, L, Nx)

# Initial temperature distribution (e.g., a Gaussian pulse in the center)
u_initial = torch.exp(-(x - L/2)**2 / 0.1)  # Gaussian function for initial temperature

# Convert to PyTorch tensor
u = u_initial.clone().detach().requires_grad_(False)  # Temperature at each spatial point
u_new = u.clone()

# Precompute constant for the finite difference method
r = alpha * dt / dx**2

# Function to update the temperature using finite differences
def heat_transfer_step(u, r):
    u_new = u.clone()
    for i in range(1, len(u) - 1):
        u_new[i] = u[i] + r * (u[i - 1] - 2 * u[i] + u[i + 1])
    return u_new

# Time-stepping loop
timesteps = int(T_max / dt)
u_all = []

# Plot initial temperature profile
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), u_initial.numpy(), label='Initial Temperature Distribution')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Initial Temperature Distribution in the Rod')
plt.grid(True)
plt.legend()
plt.show(block=True)

# Run the simulation and store results for plotting
for t in range(timesteps):
    u = heat_transfer_step(u, r)  # Update the temperature distribution
    if t % 50 == 0:  # Store every 50 steps for visualization
        u_all.append(u.clone())

# Plot the temperature distribution at different time steps
plt.figure(figsize=(10, 6))
for i, u_t in enumerate(u_all):
    plt.plot(x.numpy(), u_t.numpy(), label=f'Time = {i*50*dt:.1f} s')

plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Heat Transfer in a Rod Over Time')
plt.grid(True)
plt.legend()
plt.show(block=True)
