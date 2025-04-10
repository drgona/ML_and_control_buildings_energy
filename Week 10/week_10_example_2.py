
from neuromancer import psl
import matplotlib.pyplot as plt

"""
Simulate single zone building model using psl library in Neuromancer

See psl.systems for a full dictionary of implemented dynamical systems

Requires installation of the Neuromancer library:

    pip install neuromancer

"""

# List of available building models with nonlinear inputs
systems = ['SimpleSingleZone',      #  single zone building model
           'Reno_full',             #  six-zone residential building model with renovated envelope
           'RenoLight_full',        #  six-zone residential building model with lightweight envelope
           'Old_full',              #  six-zone residential building model with old envelope
           'HollandschHuys_full'    #  twelve-zone office building model
           ]

# List of available building models with linear inputs
systems_lin = ['LinearSimpleSingleZone', 'LinearReno_full',
               'LinearRenoLight_full', 'LinearOld_full',
               'LinearHollandschHuys_full']

"""
1, Linear state space model of the building thermal dynamics
"""
# instantiate the linear building model
system_name = "LinearReno_full"
modelSystem = psl.systems[system_name]()

# simulate the model over nsteps
nsteps = 1000
raw = modelSystem.simulate(nsim=nsteps)

# obtain simulated time series data
X = raw['X']        # latent states - temperatures of the building envelope
Y = raw['Y']        # output states - temperatures of the building zone
U = raw['U']        # inputs - heating/cooling power
D = raw['D']        # disturbances - outside temperature
T = raw['Time']     # time

# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axs[0].plot(T, Y)
axs[0].set_ylabel('Zone temperature [°C]')
axs[0].grid(True)
axs[1].plot(T, X)
axs[1].set_ylabel('Latent temperatures [°C]')
axs[1].grid(True)
axs[2].plot(T, U)
axs[2].set_ylabel('Heating/cooling power [W]')
axs[2].grid(True)
axs[3].plot(T, D)
axs[3].set_ylabel('Ambient temperature [°C]')
axs[3].set_xlabel('Time [days]')
axs[3].grid(True)
fig.suptitle('Time series of building thermal dynamics', fontsize=16)
plt.tight_layout(rect=[0., 0., 1., 1.])
plt.show(block=True)



"""
1, Linear state space model of the building thermal dynamics
with bi-linear convective heat flow equation for the HVAC system
"""
# instantiate the building model
system_name = "Reno_full"
modelSystem = psl.systems[system_name]()

# simulate the model over nsteps
nsteps = 1000
raw = modelSystem.simulate(nsim=nsteps)

# obtain simulated time series data
X = raw['X']        # latent states - temperatures of the building envelope
Y = raw['Y']        # output states - temperatures of the building zone
U = raw['U']        # inputs - mass flows and supply temperature
D = raw['D']        # disturbances - outside temperature
T = raw['Time']     # time

# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axs[0].plot(T, Y)
axs[0].set_ylabel('Zone temperature [°C]')
axs[0].grid(True)
axs[1].plot(T, X)
axs[1].set_ylabel('Latent temperatures [°C]')
axs[1].grid(True)
axs[2].plot(T, U)
axs[2].set_ylabel('Inputs: mass flow [l/h], \n supply temperature [°C]')
axs[2].grid(True)
axs[3].plot(T, D)
axs[3].set_ylabel('Ambient temperature [°C]')
axs[3].set_xlabel('Time [days]')
axs[3].grid(True)
fig.suptitle('Time series of building thermal dynamics', fontsize=16)
plt.tight_layout(rect=[0., 0., 1., 1.])
plt.show(block=True)
