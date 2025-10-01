import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

# System dynamics
A = np.array([[0.9950, 0.0017, 0.0000, 0.0031],
              [0.0007, 0.9957, 0.0003, 0.0031],
              [0.0000, 0.0003, 0.9834, 0.0000],
              [0.2015, 0.4877, 0.0100, 0.2571]])
B = np.array([[1.7586e-06],
              [1.7584e-06],
              [1.8390e-10],
              [5.0563e-04]])
C = np.array([[0.0, 0.0, 0.0, 1.0]])  # y = internal temperature

nx = A.shape[0]
nu = B.shape[1]
ny = C.shape[0]

# Output and input constraints
ymin, ymax = 20, 22
umin, umax = 0, 6000

# Weights
Qy = 1000.0
QN = Qy
R = 1e-6 * np.eye(nu)

# Setpoint and initial condition
y_ref = 21.0
x0 = 18 * np.ones(nx)
N = 10  # prediction horizon

# MPC variables
u = Variable((nu, N))       # control inputs for the prediction horizon
x_init = Parameter(nx)      # initial state for the MPC problem

objective = 0
x_k = x_init
constraints = []

for k in range(N):
    y_k = C @ x_k
    # reference tracking objective (optional: sum_squares instead of quad_form with I)
    objective += Qy * sum_squares(y_k - y_ref) + quad_form(u[:, k], R)
    # input constraints
    constraints += [umin <= u[:, k], u[:, k] <= umax]
    # dynamics rollout
    x_k = A @ x_k + B @ u[:, k]

# Terminal cost
y_k = C @ x_k
objective += QN * sum_squares(y_k - y_ref)

# Define problem
prob = Problem(Minimize(objective), constraints)

# Closed-loop simulation
nsim = 30
x_log, y_log, u_log = [x0], [(C @ x0).item()], []

for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP)
    u_applied = u[:, 0].value
    if u_applied is None:
        u_applied = np.array([0.0])
    x0 = A @ x0 + B @ u_applied
    y_applied = C @ x0

    x_log.append(x0.copy())
    y_log.append(y_applied.item())
    u_log.append(u_applied.item())

# Convert logs
x_log = np.array(x_log)
y_log = np.array(y_log)
u_log = np.array(u_log)

# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(y_log, label="T_internal (y)", linewidth=2)
plt.axhline(y_ref, color='r', linestyle='--', label='Setpoint')
plt.fill_between(np.arange(nsim+1), ymin, ymax, color='gray', alpha=0.1, label='Comfort Range')
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.title("Internal Temperature")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(u_log, label="HVAC input (Q)", linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Power Input (W)")
plt.title("HVAC input")
plt.grid()
plt.tight_layout()
plt.show(block=True)
