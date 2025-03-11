import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


# System dynamics for inverted pendulum (Cart-Pendulum system)
def pendulum_dynamics(x, u, dt):
    """
    Computes the next state of the inverted pendulum given the current state and control input.
    :param x: Current state [theta, theta_dot]
    :param u: Control input [force]
    :param dt: Time step
    :return: Next state [theta_next, theta_dot_next]
    """
    g = 9.81  # Gravity
    m = 1.0  # Mass of the pendulum
    M = 5.0  # Mass of the cart
    L = 2.0  # Length of the pendulum
    d = 1.0  # Damping coefficient

    theta = x[0]
    theta_dot = x[1]
    force = u[0]

    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)

    denominator = M + m * (1 - cos_theta ** 2)
    theta_ddot = (g * sin_theta - cos_theta * (
                force + m * L * theta_dot ** 2 * sin_theta) / denominator - d * theta_dot) / (
                             L * (4 / 3 - (m * cos_theta ** 2) / denominator))

    x_next = ca.vertcat(theta + theta_dot * dt, theta_dot + theta_ddot * dt)
    return x_next


# MPC parameters
N = 20  # Horizon length
dt = 0.1  # Time step
nx = 2  # Number of states (theta, theta_dot)
nu = 1  # Number of control inputs (force)
Q = ca.diag([10, 1])  # State cost matrix
R = ca.diag([0.1])  # Control cost matrix

# Create optimizer
opti = ca.Opti()

# Decision variables
X = opti.variable(nx, N + 1)  # States over horizon
U = opti.variable(nu, N)  # Control inputs over horizon
X0 = opti.parameter(nx)  # Initial state parameter

# Reference signal
x_ref = np.array([0, 0])  # Desired equilibrium point

# Define cost function
cost = 0
for k in range(N):
    # Quadratic cost penalizing deviation from the reference state and excessive control effort
    state_error = X[:, k] - x_ref
    control_effort = U[:, k]
    cost += state_error.T @ Q @ state_error + control_effort.T @ R @ control_effort

    # Dynamics constraint: ensure system follows the pendulum dynamics
    opti.subject_to(X[:, k + 1] == pendulum_dynamics(X[:, k], U[:, k], dt))

# Set optimization objective
opti.minimize(cost)

# Initial condition constraint
opti.subject_to(X[:, 0] == X0)

# Solver settings
opts = {'ipopt.print_level': 0, 'print_time': 0}
opti.solver('ipopt', opts)

# Simulation parameters
T = 5  # Total simulation time
steps = int(T / dt)
x_hist = []
u_hist = []

time_values = np.arange(steps) * dt
x0 = np.array([0.2, 0])  # Initial condition

# Closed-loop simulation
for t in range(steps):
    opti.set_value(X0, x0)  # Set current state as initial condition
    sol = opti.solve()  # Solve the optimization problem

    sol_X = sol.value(X)  # Get optimized state trajectory
    sol_U = sol.value(U).flatten()  # Get optimized control sequence

    u_opt = sol_U[0]  # Apply the first control input
    x0 = np.array(pendulum_dynamics(x0, [u_opt], dt)).flatten()  # Compute next state

    x_hist.append(x0)
    u_hist.append(u_opt)

# Convert results to numpy arrays
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(time_values, x_hist[:, 0], label='Theta (rad)')
ax[0].plot(time_values, x_hist[:, 1], label='Theta_dot (rad/s)')
ax[0].plot(time_values, np.full_like(time_values, x_ref[0]), 'r--', label='Reference Theta')
ax[0].plot(time_values, np.full_like(time_values, x_ref[1]), 'r--', label='Reference Theta_dot')
ax[0].set_ylabel('State values')
ax[0].legend()
ax[0].grid()

ax[1].plot(time_values, u_hist, label='Control Input (Force)')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Force (N)')
ax[1].legend()
ax[1].grid()

plt.show(block=True)
