import numpy as np

from integrators import rk4
from models import pendulum

TIMESTEP = 0.001
NUM_STEPS = 100


def simulate(params, initial_state):
    """Return a (2, NUM_STEPS + 1) trajectory integrated with RK4."""
    state_traj = np.zeros((2, NUM_STEPS + 1))
    state_traj[:, 0] = initial_state
    for step in range(NUM_STEPS):
        state_traj[:, step + 1] = rk4(
            pendulum.dynamics, step * TIMESTEP, state_traj[:, step], TIMESTEP, params
        )
    return state_traj


def total_energy(state_traj, params):
    potential_energy, kinetic_energy = pendulum.calculate_energy(state_traj, params)
    return potential_energy + kinetic_energy


def test_energy_conserved_without_damping_or_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    state_traj = simulate(params, initial_state=np.array([0.5, 0.0]))

    energy = total_energy(state_traj, params)
    assert np.all(np.isclose(np.diff(energy), 0.0, atol=1e-8))


def test_torque_does_work():
    """A constant torque adds energy torque * (change in angle), with its sign."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0
    state_traj = simulate(params, initial_state=np.array([0.5, 0.0]))

    energy = total_energy(state_traj, params)
    energy_change = energy - energy[0]
    work = params["torque"] * (state_traj[0] - state_traj[0, 0])
    assert np.all(np.isclose(energy_change, work, atol=1e-8))


def test_damping_dissipates_energy():
    """Damping never adds energy and removes damping_coeff * integral(omega^2 dt)."""
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.5
    params["torque"] = 0.0
    state_traj = simulate(params, initial_state=np.array([0.5, 1.0]))

    energy = total_energy(state_traj, params)
    assert np.all(np.diff(energy) <= 0.0)
    omega_squared = state_traj[1] ** 2
    dissipated = params["damping_coeff"] * np.trapezoid(omega_squared, dx=TIMESTEP)
    assert np.isclose(energy[0] - energy[-1], dissipated, rtol=1e-3)
