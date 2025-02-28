import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

"""
Linear State-Space System Identification using Least Squares

"""

# Generate synthetic data for a linear state-space system
def generate_linear_state_space_data(steps=1000):
    A_true = np.array([[0.9, 0.2], [0.1, 0.7]])  # True state transition matrix
    B_true = np.array([[0.1], [0.05]])  # True input matrix
    C_true = np.array([[1.0, 0.0]])  # Output matrix
    D_true = np.array([[0.0]])  # Direct feedthrough matrix

    x = np.zeros((steps, 2))  # State vector (x1, x2)
    u = np.sin(np.linspace(0, 10, steps)).reshape(-1, 1)  # Input signal
    y = np.zeros((steps, 1))  # Output

    for t in range(1, steps):
        x[t] = A_true @ x[t - 1] + B_true @ u[t - 1] + 0.01 * np.random.randn(2)  # Process noise
        y[t] = C_true @ x[t] + D_true @ u[t] + 0.1 * np.random.randn(1)  # Measurement noise

    return u, y, x


# Least Squares System Identification
def least_squares_state_space(U, Y, X):
    """
    Estimate A, B, C, and D using Least Squares from collected data.
    """

    # Prepare data using single step time shift
    X_past, X_future = X[:-1, :], X[1:, :]
    U_past = U[:-1, :]
    Y_measured = Y[:-1, :]

    # Ensure consistent dimensions before concatenation
    X_past = X_past.reshape(X_past.shape[0], -1)
    X_future = X_future.reshape(X_future.shape[0], -1)
    U_past = U_past.reshape(U_past.shape[0], -1)
    Y_measured = Y_measured.reshape(Y_measured.shape[0], -1)

    # Solve for A and B using least squares: X_future = [A B] * [X_past U_past]
    AB, _, _, _ = lstsq(np.hstack([X_past, U_past]), X_future)
    A_est, B_est = AB[:X.shape[1], :].T, AB[X.shape[1]:, :].T

    # Solve for C and D using least squares: Y = [C D] * [X U]
    CD, _, _, _ = lstsq(np.hstack([X_past, U_past]), Y_measured)
    C_est, D_est = CD[:X.shape[1], :].T, CD[X.shape[1]:, :].T

    return A_est, B_est, C_est, D_est


# Generate data
U, Y, X = generate_linear_state_space_data()

# Identify system matrices using least squares
A_est, B_est, C_est, D_est = least_squares_state_space(U, Y, X)

# Print estimated system matrices
print("Estimated A matrix:\n", A_est)
print("Estimated B matrix:\n", B_est)
print("Estimated C matrix:\n", C_est)
print("Estimated D matrix:\n", D_est)

# Simulate the identified model
X_pred = [np.zeros((2, 1))]
Y_pred = []
for t in range(len(U)):
    U_t = U[t].reshape(-1, 1)  # Ensure input is of shape (input_dim, 1)
    X_next = A_est @ X_pred[-1] + B_est @ U_t
    Y_next = C_est @ X_pred[-1] + D_est @ U_t
    X_pred.append(X_next)
    Y_pred.append(Y_next)

Y_pred = np.array(Y_pred).squeeze()

# Plot results
plt.figure(figsize=(6, 4))
plt.plot(Y, label='True Output', alpha=0.6)
plt.plot(Y_pred, label='Predicted Output', linestyle='dashed')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('Linear State-Space System Identification (Least Squares)')
plt.grid()
plt.show(block=True)
