import torch
import matplotlib.pyplot as plt

"""
Example 2: Solve initial value problem (IVP) of a non-linear ODE

Solver: Euler method 
ODE: two tank system

"""

# Define the nonlinear ODE system for a two-tank system
def two_tank_system(state):
    h1, h2 = state
    a1, a2, g = 0.5, 0.5, 9.81  # Tank parameters
    dh1dt = -a1 * torch.sqrt(2 * g * h1) + a2 * torch.sqrt(2 * g * h2)
    dh2dt = a1 * torch.sqrt(2 * g * h1) - a2 * torch.sqrt(2 * g * h2)
    return torch.tensor([dh1dt, dh2dt], dtype=torch.float32)

# Parameters
h0 = torch.tensor([1.0, 0.5], dtype=torch.float32)  # Initial condition
N = 100 # simulation steps
dt = 0.1 # Time step
t0 = 0.0  # initial time

# Solve using Euler's method
state_values = [h0]
time = [t0]
for i in range(1, N):
    state_next = state_values[-1] + two_tank_system(state_values[-1]) * dt
    state_values.append(state_next)
    t_next = time[-1] + dt
    time.append(t_next)

# Convert to tensor for plotting
state_values = torch.stack(state_values).detach().numpy()

# Plot time series
plt.figure(figsize=(10, 5))
plt.plot(time, state_values[:, 0], label='h1(t)', color='b')
plt.plot( time, state_values[:, 1], label='h2(t)', color='r')
plt.xlabel('Time')
plt.ylabel('Tank Levels')
plt.title('Time Series Plot')
plt.legend()
plt.grid()
plt.show(block=True)

