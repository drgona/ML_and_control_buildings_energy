import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
Example 3

Rule-based control of a simple model of a thermal dynamics of a house

system: 1D linear ordinary differential equation 
controller: relay thermostat

"""

# Define the temperature dynamics model with a relay thermostat
def temperature_dynamics(T, t, C, R, T_outside, T_set, delta_T, P_heater):
    """ Differential equation for indoor temperature dynamics with a relay thermostat """

    # Relay thermostat control logic
    global heater_status  # Track heater state
    if T < T_set - delta_T:
        heater_status = True  # Heater ON
    elif T > T_set + delta_T:
        heater_status = False  # Heater OFF

    # Apply heating power if heater is ON
    if heater_status:
        P = P_heater
    else:
        P = 0

    # Temperature change rate equation
    dTdt = (-1 / (R * C)) * (T - T_outside) + P / C
    return dTdt


# Parameters
C = 100  # Thermal capacity of the house (Joule/°C)
R = 50  # Thermal resistance (°C/Watt)
T_outside = -5  # Outdoor temperature (°C)
P_heater = 10  # Heater power (Watt)

# Thermostat setpoint and hysteresis
T_set = 20  # Desired temperature (°C)
delta_T = 1  # Hysteresis band (°C)

# Initial conditions
T0 = 15  # Initial indoor temperature (°C)
heater_status = False  # Initial heater state (OFF)

# Time simulation
t = np.linspace(0, 6 * 3600, 1000)  # Simulating for 12 hours (in seconds)

# Solve the system
T_solution = odeint(temperature_dynamics, T0, t, args=(C, R, T_outside, T_set, delta_T, P_heater))

# Convert time to hours for better readability
t_hours = t / 3600  # Convert seconds to hours

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t_hours, T_solution, label="Indoor Temperature", color='b')
plt.axhline(y=T_set, color='r', linestyle='--', label="Thermostat Setpoint")
plt.axhline(y=T_set - delta_T, color='orange', linestyle=':', label="Lower Bound (Heater ON)")
plt.axhline(y=T_set + delta_T, color='orange', linestyle=':', label="Upper Bound (Heater OFF)")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.title("House Temperature Dynamics with Relay Thermostat")
plt.legend()
plt.grid(True)
plt.show(block=True)
