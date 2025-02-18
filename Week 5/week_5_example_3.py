import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate synthetic data for a quadratic surface: Y = 3X1² + 2X2² + 4X1X2 + 5X1 + 6X2 + 7
np.random.seed(42)
X1 = np.linspace(0, 10, 50)
X2 = np.linspace(0, 10, 50)
X1, X2 = np.meshgrid(X1, X2)

# Define polynomial equation with some small noise
Y = 3 * X1**2 + 2 * X2**2 + 4 * X1 * X2 + 5 * X1 + 6 * X2 + 7 + np.random.randn(*X1.shape) * 3

# Reshape data for fitting
X_train = np.column_stack((X1.ravel(), X2.ravel()))
Y_train = Y.ravel()

# Apply polynomial feature transformation (degree=2 for quadratic terms)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, Y_train)

# Predict Y values using the trained model
Y_pred = model.predict(X_poly)

# Reshape predictions to match the meshgrid shape
Y_pred = Y_pred.reshape(X1.shape)

# Print learned coefficients
print("Fitted Polynomial Equation Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")

# Plot the original points and fitted polynomial surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual data
ax.scatter(X1, X2, Y, color='red', label="Actual Data", alpha=0.5)

# Plot the fitted polynomial surface
ax.plot_surface(X1, X2, Y_pred, color='cyan', alpha=0.6, edgecolor='black')

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Polynomial Regression Surface (Quadratic)')

plt.legend()
plt.show(block=True)
