import numpy as np
import matplotlib.pyplot as plt

"""
Model Predictive Path Integral (MPPI) Control for the building thermal model 

MPPI Control is a powerful sampling-based stochastic control method — 
a great alternative to traditional MPC, 
especially when dealing with nonlinearities or uncertainty
"""

# System matrices (same as before)
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
C = np.array([[0, 0, 0, 1]])

nx = A.shape[0]
nu = B.shape[1]
nd = E.shape[1]
ny = C.shape[0]

# Simulation and control parameters
N = 15                    # Horizon
K = 100                   # Number of sampled trajectories
lambda_ = 1.0             # Temperature for softmax weighting
sigma = 500.0             # Std deviation of control noise
Qy = 10.0                 # Output tracking cost
Ru = 1e-6                 # Control energy cost

# Initial state and reference
x0 = 18 * np.ones(nx)
y_ref = 21.0
umin, umax = 0, 6000

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


# Logs
x_log, y_log, u_log = [x0], [(C @ x0).item()], []


# MPPI loop in closed-loop simulation
x = x0.copy()
for t0 in range(nsim):
    d_window = d_sequence[:, t0:t0+N]
    u_nominal = np.full((nu, N), 1000.0)  # nominal input sequence

    # Sample K trajectories with noise
    costs = np.zeros(K)
    U_samples = np.zeros((K, nu, N))
    for k in range(K):
        u_noise = sigma * np.random.randn(nu, N)
        u_traj = np.clip(u_nominal + u_noise, umin, umax)
        U_samples[k] = u_traj

        # Simulate N-step long rollout (model prediction) for each sampled control sequence
        x_sim = x.copy()
        cost = 0.0
        for i in range(N):
            u = u_traj[:, i]
            d = d_window[:, i]
            y = C @ x_sim
            cost += Qy * (y - y_ref) ** 2 + Ru * np.sum(u ** 2)
            x_sim = A @ x_sim + B @ u + E @ d
        costs[k] = cost

    # Path integral weighting
    beta = np.min(costs)
    weights = np.exp(-(costs - beta) / lambda_)
    weights /= np.sum(weights)

    # Compute weighted average control
    u_mppi = np.sum(weights[:, None, None] * U_samples, axis=0)
    u_applied = u_mppi[:, 0]    # receding horizon control

    # Apply control
    x = A @ x + B @ u_applied + E @ d_sequence[:, t0]
    y = C @ x

    x_log.append(x.copy())
    y_log.append(y.item())
    u_log.append(u_applied.item())

# Convert logs
x_log = np.array(x_log)
y_log = np.array(y_log)
u_log = np.array(u_log)

# Plot results
plt.figure(figsize=(12, 7))

plt.subplot(3, 1, 1)
plt.plot(y_log, label="Internal Temp (y)", linewidth=2)
plt.axhline(y_ref, color='r', linestyle='--', label='Setpoint')
plt.fill_between(np.arange(nsim+1), 20, 22, color='gray', alpha=0.1, label='Comfort Zone')
plt.ylabel("Temperature (°C)")
plt.title("MPPI-Controlled Internal Temperature")
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(u_log, label="HVAC input (Q)", linewidth=2)
plt.ylabel("Power (W)")
plt.title("Control Input (HVAC Power)")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
labels = ["External Temp", "Occupancy", "Solar"]
for i in range(3):
    plt.plot(d_sequence[i, :nsim+1], label=labels[i])
plt.xlabel("Time Step")
plt.ylabel("Disturbance")
plt.title("Disturbances")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show(block=True)
