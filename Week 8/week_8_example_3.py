import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Discretization settings
dt = 0.1  # Time step
N = 20  # Prediction horizon (short-term NMPC)
T_sim = 100  # Total simulation time (longer)

# System parameters
A = 1.0  # Cross-sectional area of tanks
a1 = 0.1  # Outlet valve area of Tank 1
a2 = 0.1  # Outlet valve area of Tank 2
g = 9.81  # Gravity

# State and control bounds
h_max = 2.0  # Maximum tank level
q_max = 1.0  # Maximum pump flow rate
h_min = 1e-3  # Min level to prevent sqrt issues
q_min = 1e-3  # Min inflow to prevent drying out


# Define step changes in reference signal
def step_reference(t):
    if t < 20:
        return 0.5
    elif t < 50:
        return 1.2
    elif t < 80:
        return 0.8
    else:
        return 1.5


# Initialize state
h1 = 1.0  # Initial water level in Tank 1
h2 = 0.5  # Initial water level in Tank 2
x0 = np.array([h1, h2])

# Storage for simulation results
time_sim = np.arange(0, T_sim, dt)
h1_history = []
h2_history = []
q_in_history = []
ref_history = []



# Run NMPC in closed-loop simulation
for t in time_sim:
    # CasADi optimization setup
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(2, N + 1)  # States: [h1, h2]
    U = opti.variable(1, N)  # Control input (inflow to Tank 1)

    # Reference trajectory over the prediction horizon
    h2_ref = np.array([step_reference(t + k * dt) for k in range(N + 1)])

    # Cost function
    lambda_u = 0.01  # Regularization weight for control effort
    J = 0
    for k in range(N):
        J += (X[1, k] - h2_ref[k]) ** 2 + lambda_u * U[0, k] ** 2

    opti.minimize(J)

    # System dynamics constraints using Euler discretization
    for k in range(N):
        h1_k = X[0, k]
        h2_k = X[1, k]
        q_in_k = U[0, k]

        # Ensure non-negative levels for square root calculations
        h1_safe = ca.fmax(h1_k, h_min)
        h2_safe = ca.fmax(h2_k, h_min)

        # Compute flow rates
        q1_out = a1 * ca.sqrt(2 * g * h1_safe)
        q2_out = a2 * ca.sqrt(2 * g * h2_safe)

        # Discrete-time dynamics
        h1_next = h1_k + dt * (q_in_k - q1_out) / A
        h2_next = h2_k + dt * (q1_out - q2_out) / A

        # Constraint: Next state must match dynamics
        opti.subject_to(X[:, k + 1] == ca.vertcat(h1_next, h2_next))

    # Constraints: Initial condition
    opti.subject_to(X[:, 0] == x0)

    # Constraints: State and control limits (element-wise)
    opti.subject_to(h_min <= X[0, :])
    opti.subject_to(X[0, :] <= h_max)

    opti.subject_to(h_min <= X[1, :])
    opti.subject_to(X[1, :] <= h_max)

    # Control constraints
    opti.subject_to(q_min <= U)
    opti.subject_to(U <= q_max)

    # Solver settings
    opts = {
        'ipopt': {
            'tol': 1e-6,
            'print_level': 0,
            'max_iter': 500,
            'acceptable_tol': 1e-5,
            'acceptable_iter': 10
        }
    }
    opti.solver('ipopt', opts)

    # Solve NMPC
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"Solver failed at time {t}: {e}")
        break

    # Extract first control action and update system state
    q_in_opt = sol.value(U[0, 0])
    h1_opt = sol.value(X[0, 1])
    h2_opt = sol.value(X[1, 1])

    # Store results
    h1_history.append(h1_opt)
    h2_history.append(h2_opt)
    q_in_history.append(q_in_opt)
    ref_history.append(step_reference(t))

    # Update initial state for next iteration (simulate system)
    x0 = np.array([h1_opt, h2_opt])

# Convert to arrays for plotting
h1_history = np.array(h1_history)
h2_history = np.array(h2_history)
q_in_history = np.array(q_in_history)
ref_history = np.array(ref_history)

# Plot results
plt.figure(figsize=(12, 6))

# Tank levels
plt.subplot(2, 1, 1)
plt.plot(time_sim[:len(h1_history)], h1_history, label="Tank 1 Level", color='b')
plt.plot(time_sim[:len(h2_history)], h2_history, label="Tank 2 Level", color='g')
plt.plot(time_sim[:len(ref_history)], ref_history, 'r--', label="Tank 2 Reference")
plt.xlabel("Time [s]")
plt.ylabel("Tank Levels [m]")
plt.legend()
plt.grid()

# Control input
plt.subplot(2, 1, 2)
plt.step(time_sim[:len(q_in_history)], q_in_history, label="Pump Flow Rate", where='post', color='purple')
plt.xlabel("Time [s]")
plt.ylabel("Flow Rate [m³/s]")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)
