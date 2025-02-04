import torch
import matplotlib.pyplot as plt

# Define the Lorenz system
def lorenz_system(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return torch.tensor([dxdt, dydt, dzdt], dtype=torch.float32)

# Runge-Kutta 4th order method
def runge_kutta_step(f, state, dt):
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Parameters
initial_state = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
dt = 0.01
N = 10000

# Solve the Lorenz system using Runge-Kutta method
state_values = [initial_state]
for _ in range(N):
    next_state = runge_kutta_step(lorenz_system, state_values[-1], dt)
    state_values.append(next_state)

# Convert to tensor for plotting
state_values = torch.stack(state_values).detach().numpy()

# Plot the results
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_values[:, 0], state_values[:, 1], state_values[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz System Trajectory')
plt.show(block=True)