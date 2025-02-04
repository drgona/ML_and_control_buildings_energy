import torch
import matplotlib.pyplot as plt

"""
Example 3: Linearizing a non-linear ODE to obtain state space model (SSM)

Solver: matrix exponential using Jacobian linearization at each step
ODE: two-tank system

"""


# constants related to the system's geometry (tank size and outflow characteristics)
a1, a2 = 0.5, 0.5
# gravitational constant
g = 9.81

# Define the nonlinear ODE system for a two-tank system
def two_tank_system(state):
    h1, h2 = state
    dh1dt = -a1 * torch.sqrt(2 * g * h1) + a2 * torch.sqrt(2 * g * h2)
    dh2dt = a1 * torch.sqrt(2 * g * h1) - a2 * torch.sqrt(2 * g * h2)
    return torch.tensor([dh1dt, dh2dt], dtype=torch.float32)


# Compute the Jacobian matrix for linearization
def jacobian_matrix(h1, h2):
    # Compute the partial derivatives
    j11 = -a1 / (2 * torch.sqrt(2 * g * h1 + 1e-6))  # Avoid division by zero
    j12 = a2 / (2 * torch.sqrt(2 * g * h2 + 1e-6))
    j21 = a1 / (2 * torch.sqrt(2 * g * h1 + 1e-6))
    j22 = -a2 / (2 * torch.sqrt(2 * g * h2 + 1e-6))
    return torch.tensor([[j11, j12], [j21, j22]], dtype=torch.float32)


# Simulation parameters
h0 = torch.tensor([1.0, 0.5], dtype=torch.float32)  # Initial condition
N = 1000  # simulation steps
dt = 0.1  # Time step
t0 = 0.0  # initial time

# Solve using matrix exponential (Linearized system)
state_values = [h0]
time = [t0]
for i in range(1, N):
    # Get the current state
    current_state = state_values[-1]
    h1, h2 = current_state[0], current_state[1]

    # Compute Jacobian matrix at the current initial state h0
    A = jacobian_matrix(h1, h2)

    # Compute matrix exponential: e^(A * dt)
    exp_A_dt = torch.matrix_exp(A * dt)

    # Compute the next state: state_next = exp(A * dt) * state
    state_next = torch.matmul(exp_A_dt, current_state)

    # Append the state and time to their respective lists
    state_values.append(state_next)
    # integrate time
    t_next = time[-1] + dt
    time.append(t_next)

# Convert state_values to tensor for plotting
state_values = torch.stack(state_values).detach().numpy()

# Plot time series
plt.figure(figsize=(10, 5))
plt.plot(time, state_values[:, 0], label='h1(t)', color='b')
plt.plot(time, state_values[:, 1], label='h2(t)', color='r')
plt.xlabel('Time')
plt.ylabel('Tank Levels')
plt.title('Time Series Plot')
plt.legend()
plt.grid()
plt.show(block=True)
