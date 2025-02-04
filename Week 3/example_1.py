import torch
import matplotlib.pyplot as plt

"""
Example 1: Solve initial value problem (IVP) of a linear ODE

Solver: Euler method 
ODE: simple exponential growth

"""


# Define the exponential growth ODE
def exponential_growth(x, r):
    return r * x

# Parameters
r = 0.5  # Growth rate
x0 = torch.tensor([1.0], dtype=torch.float32)  # Initial condition
dt = 0.1 # Time step
N = 100 # simulation steps

# Solve using Euler's method
x_values = [x0]
for i in range(1, N):
    x_next = x_values[-1] + dt * exponential_growth(x_values[-1], r)
    x_values.append(x_next)

# Convert to tensor for plotting
x_values = torch.stack(x_values).detach().numpy()

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_values, label='Exponential Growth', color='b')
plt.xlabel('Time steps')
plt.ylabel('x(t)')
plt.title('Exponential Growth ODE Solution')
plt.legend()
plt.grid()
plt.show(block=True)
