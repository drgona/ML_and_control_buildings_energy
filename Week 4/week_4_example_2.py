import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


"""
Example 2

Simple model of a thermal dynamics of a house

system: 1D linear ordinary differential equation 

"""

# Define the temperature dynamics model
def temperature_dynamics(T, t, C, R, T_outside):
    """ Differential equation for indoor temperature dynamics """

    # Temperature change rate equation
    dTdt = (-1 / (R*C)) * (T - T_outside)
    return dTdt

# Parameters
C = 100  # Thermal capacity of the house (Joule/°C)
R = 50    # Thermal resistance (°C/Watt)
T_outside = 15  # Outdoor temperature (°C)
P_heater = 0  # Heater power (Watt)

# Initial condition
T0 = 20  # Initial indoor temperature (°C)

# Time simulation
t = np.linspace(0, 24 * 3600, 1000)  # Simulating for 24 hours (in seconds)

# Solve the system
T_solution = odeint(temperature_dynamics, T0, t, args=(C, R, T_outside))

# Convert time to hours for better readability
t_hours = t / 3600  # Convert seconds to hours

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t_hours, T_solution, label="Indoor Temperature", color='b')
plt.axhline(y=T_outside, color='gray', linestyle=':', label="Outdoor Temperature")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.title("House Temperature Dynamics")
plt.legend()
plt.grid(True)
plt.show(block=True)
