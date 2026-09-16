import numpy as np
import matplotlib.pyplot as plt

from models import bouncing_ball as model
from integrators import rk4 as integrator

params = model.generate_params()

# some set-up
initial_state = np.array([5.0, 0.0])  # start at height 5m, zero velocity

timestep = 1e-3
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
for step, t in enumerate(time_traj[:-1]):
    next_state = integrator(
        model.dynamics, t, state_traj[:, step], params, timestep
    )
    next_state = model.apply_bounce(next_state, params)  # <-- collision check
    state_traj[:, step + 1] = next_state

# sanity check the energies
kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
total_energy = kinetic_energy + potential_energy

plt.figure()
plt.plot(time_traj, state_traj[0, :])
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Bouncing ball height")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, total_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing ball energy")
plt.legend()
plt.tight_layout()
plt.show()