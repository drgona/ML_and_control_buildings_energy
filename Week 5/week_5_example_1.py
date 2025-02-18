import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable
Y = np.array([2, 4, 6, 8, 10])  # Dependent variable (Y = 2X)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Get the learned parameters
A = model.coef_[0]  # Slope
b = model.intercept_  # Intercept

print(f"Trained Model: Y = {A:.2f}X + {b:.2f}")

# Make predictions
X_test = np.array([6, 7]).reshape(-1, 1)
Y_pred = model.predict(X_test)
print(f"Predictions for X = [6, 7]: {Y_pred}")

# Plot results
plt.scatter(X, Y, color='blue', label="Original Data")
plt.plot(X, model.predict(X), color='red', label="Fitted Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show(block=True)

