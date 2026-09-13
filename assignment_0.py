import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import pendulum as model
from integrators import rk4 as integrator
# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 1e-2
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

def sim():
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator(
            model.dynamics, t, state_traj[:, step], params, timestep
        )

elapsed = timeit.timeit(sim, number=20)
avg_time = elapsed / 20

print(f"Av execution time: {avg_time:.6f}s")


# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

potential_energy, kinetic_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

# Phase portrait plot
plt.figure()
plt.plot(state_traj[0, :], state_traj[1, :], label="Phase Trajectory")
plt.scatter(initial_state[0], initial_state[1], color="red", zorder=5, label="Start")
plt.xlabel("Position $\\theta$ (rad) x")
plt.ylabel("Velocity $\\dot{\\theta}$ (rad/s) f(x)")
plt.title("Pendulum Phase Portrait")
plt.legend()
plt.tight_layout()
plt.show()