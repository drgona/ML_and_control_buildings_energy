import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

"""
Linear MPC of a simple building thermal model

Reference tracking formulation 
dynamics constraints given in a dense form (single shooting)
dynamics with disturbances (external temperature, occupancy, solar gain)

"""

# System dynamics
A = np.array([[0.9950, 0.0017, 0.0000, 0.0031],
              [0.0007, 0.9957, 0.0003, 0.0031],
              [0.0000, 0.0003, 0.9834, 0.0000],
              [0.2015, 0.4877, 0.0100, 0.2571]])
B = np.array([[1.7586e-06],
              [1.7584e-06],
              [1.8390e-10],
              [5.0563e-04]])
E = np.array([[0.0002, 0.0000, 0.0000],
              [0.0002, 0.0000, 0.0000],
              [0.0163, 0.0000, 0.0000],
              [0.0536, 0.0005, 0.0001]])
C = np.array([[0.0, 0.0, 0.0, 1.0]])

nx = A.shape[0]
nu = B.shape[1]
nd = E.shape[1]
ny = C.shape[0]

# Constraints
ymin, ymax = 20, 22
umin, umax = 0, 6000

# Weights
Qy = 1000.0
QN = Qy
R = 1e-6 * np.eye(nu)

# Setpoint
y_ref = 21.0
x0 = 18 * np.ones(nx)
N = 10  # prediction horizon

# Simulated disturbances for 40 steps (nsim + N)
np.random.seed(0)
nsim = 30
# Time base
t = np.linspace(0, 4 * np.pi, nsim + N)
# More complex and realistic external temperature variation
daily_variation = 8 * np.sin(t)               # daily sinusoidal swings
noise = 1.5 * np.random.randn(nsim + N)       # random weather noise
trend = -0.03 * np.arange(nsim + N)           # slight cooling trend

d_sequence = np.zeros((nd, nsim + N))  # (external T, occupancy, solar)
d_sequence[0, :] = 28 + daily_variation + noise + trend  # external temp
d_sequence[1, :] = 100 * (np.random.rand(nsim + N) > 0.8)  # occupancy spikes
d_sequence[2, :] = 500 * np.clip(np.sin(np.linspace(0, 4*np.pi, nsim + N)), 0, 1)  # solar


# MPC variables
u = Variable((nu, N))       # control inputs for the prediction horizon
s_y = Variable((ny, N), nonneg=True) # slack for y constraints (for soft constraints)
x_init = Parameter(nx)  # initial state for the MPC problem
d_param = Parameter((nd, N))  # disturbance forecast

objective = 0
x_k = x_init
constraints = []

for k in range(N):
    d_k = d_param[:, k]
    y_k = C @ x_k
    objective += Qy * quad_form(y_k - y_ref, np.eye(ny)) + quad_form(u[:, k], R) + 1e6 * sum(s_y[:, k])
    constraints += [y_k >= ymin - s_y[:, k], y_k <= ymax + s_y[:, k]]
    constraints += [umin <= u[:, k], u[:, k] <= umax]
    x_k = A @ x_k + B @ u[:, k] + E @ d_k

# Terminal cost
y_k = C @ x_k
objective += QN * quad_form(y_k - y_ref, np.eye(ny))

# Define problem
prob = Problem(Minimize(objective), constraints)

# Closed-loop simulation
x_log, y_log, u_log = [x0], [(C @ x0).item()], []

for i in range(nsim):
    x_init.value = x0
    d_window = d_sequence[:, i:i+N]
    d_param.value = d_window

    prob.solve(solver=SCS)
    u_applied = u[:, 0].value
    if u_applied is None:
        u_applied = np.array([0.0])

    x0 = A @ x0 + B @ u_applied + E @ d_sequence[:, i]
    y_applied = C @ x0

    x_log.append(x0.copy())
    y_log.append(y_applied.item())
    u_log.append(u_applied.item())

# Convert logs
x_log = np.array(x_log)
y_log = np.array(y_log)
u_log = np.array(u_log)

# Plot results
plt.figure(figsize=(12, 8))

# Output (temperature)
plt.subplot(3, 1, 1)
plt.plot(y_log, label="T_internal (y)", linewidth=2)
plt.axhline(y_ref, color='r', linestyle='--', label='Setpoint')
plt.fill_between(np.arange(nsim+1), ymin, ymax, color='gray', alpha=0.1, label='Comfort Range')
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.title("Internal Temperature with Disturbance")
plt.grid()
plt.legend()

# Input (HVAC power)
plt.subplot(3, 1, 2)
plt.plot(u_log, label="HVAC input (Q)", linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Power Input (W)")
plt.title("HVAC Input Over Time")
plt.grid()
plt.legend()

# Disturbances
plt.subplot(3, 1, 3)
labels = ["External Temp (°C)", "Occupancy Gain", "Solar Gain"]
colors = ["tab:blue", "tab:orange", "tab:green"]
for i in range(3):
    plt.plot(d_sequence[i, :nsim+1], label=labels[i], linewidth=2, color=colors[i])
plt.xlabel("Time Step")
plt.ylabel("Disturbance Value")
plt.title("Disturbances Over Time")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show(block=True)
