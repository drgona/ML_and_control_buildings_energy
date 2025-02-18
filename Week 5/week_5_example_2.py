import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data that fits a plane Y = 3X1 + 2X2 + 5
np.random.seed(42)
X1 = np.linspace(0, 10, 100)  # 100 points for X1
X2 = np.linspace(0, 10, 100)  # 100 points for X2

# Create a grid of values
X1, X2 = np.meshgrid(X1, X2)

# Define the plane equation with some small noise
Y = 3 * X1 + 2 * X2 + 5 + np.random.randn(*X1.shape) * 1.5

# Reshape data for fitting
X_train = np.column_stack((X1.ravel(), X2.ravel()))
Y_train = Y.ravel()

# Train linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predicted Y values using the model
Y_pred = model.predict(X_train)

# Reshape predictions to match the meshgrid shape
Y_pred = Y_pred.reshape(X1.shape)

# Print model parameters
print("Fitted Plane Equation:")
print(f"Y = {model.coef_[0]:.2f}*X1 + {model.coef_[1]:.2f}*X2 + {model.intercept_:.2f}")

# Plot the original points and fitted plane
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter original data points
ax.scatter(X1, X2, Y, color='red', label="Actual Data", alpha=0.5)

# Plot the fitted plane
ax.plot_surface(X1, X2, Y_pred, color='cyan', alpha=0.6)

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Fitted Plane using Multivariate Linear Regression')

plt.legend()
plt.show(block=True)
