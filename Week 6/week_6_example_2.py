import torch
import matplotlib.pyplot as plt
import numpy as np

"""
Dynamical system excitation 

response of two tank system to step and PRBS input signals

"""


# Define system parameters
A = 1.0  # Cross-sectional area of tanks
a1 = 0.1  # Valve coefficient between tanks
a2 = 0.1  # Valve coefficient for tank 2 outflow
g = 9.81  # Gravitational constant

# Time parameters
dt = 0.1  # Time step
T = 100  # Total simulation time
steps = int(T / dt)

# Step input
q_in_step = torch.zeros(steps)
q_in_step[100:600] = 0.5  # Step input applied at step 100
q_in_step[600:] = 0.2  # Step input applied at step 600

# PRBS input (Pseudo-Random Binary Sequence)
np.random.seed(42)
prbs_signal = np.random.choice([0.0, 0.5], size=steps, p=[0.5, 0.5])
q_in_prbs = torch.tensor(prbs_signal, dtype=torch.float32)

# Initialize tank levels
h1_step = torch.zeros(steps)
h2_step = torch.zeros(steps)
h1_prbs = torch.zeros(steps)
h2_prbs = torch.zeros(steps)


def two_tank_system(q_in, h1, h2):
    dh1 = (1 / A) * (q_in - a1 * torch.sqrt(2 * g * h1))
    dh2 = (1 / A) * (a1 * torch.sqrt(2 * g * h1) - a2 * torch.sqrt(2 * g * h2))
    return dh1, dh2


# Simulation using Euler's method
for t in range(steps - 1):
    # Step Response
    dh1_step, dh2_step = two_tank_system(q_in_step[t], h1_step[t], h2_step[t])
    h1_step[t + 1] = h1_step[t] + dt * dh1_step
    h2_step[t + 1] = h2_step[t] + dt * dh2_step

    # PRBS Response
    dh1_prbs, dh2_prbs = two_tank_system(q_in_prbs[t], h1_prbs[t], h2_prbs[t])
    h1_prbs[t + 1] = h1_prbs[t] + dt * dh1_prbs
    h2_prbs[t + 1] = h2_prbs[t] + dt * dh2_prbs

# Plot system response and input signal in subplots
plt.figure(figsize=(8, 8))


# System step response plot
plt.subplot(2, 1, 1)
plt.plot(torch.arange(steps) * dt, h1_step, label='Tank 1 Level (step)', linewidth=2)
plt.plot(torch.arange(steps) * dt, h2_step, label='Tank 2 Level (step)', linewidth=2)
plt.ylabel('Water Level (m)')
plt.title('Step Response of Two-Tank System')
plt.legend()
plt.grid()

# Input signal plot
plt.subplot(2, 1, 2)
plt.plot(torch.arange(steps) * dt, q_in_step, label='Input Flow Rate', linewidth=2, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Flow Rate (m^3/s)')
plt.title('Input Signal (Step)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)


plt.figure(figsize=(8, 8))
# PRBS Response Plot
plt.subplot(2, 1, 1)
plt.plot(torch.arange(steps) * dt, h1_prbs, label='Tank 1 Level (PRBS)', linewidth=2)
plt.plot(torch.arange(steps) * dt, h2_prbs, label='Tank 2 Level (PRBS)', linewidth=2)
plt.ylabel('Water Level (m)')
plt.title('PRBS Response of Two-Tank System')
plt.legend()
plt.grid()

# Input Signal Plot
plt.subplot(2, 1, 2)
plt.plot(torch.arange(steps) * dt, q_in_prbs, label='PRBS Input', linewidth=2, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Flow Rate (m^3/s)')
plt.title('Input Signals (PRBS)')
plt.legend()
plt.show(block=True)