import numpy as np


def dynamics(t, state, params):
    """Free-fall dynamics: no collision handling here, just gravity."""
    gravity = params["gravity"]

    height = state[0]
    velocity = state[1]

    acceleration = -gravity

    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "restitution": 1,  # coefficient of restitution
    }
    return params


def apply_bounce(state, params):
    """Check for floor collision and reflect velocity if needed.

    Call this after every integrator step, on top of the free-fall dynamics.
    """
    height = state[0]
    velocity = state[1]
    restitution = params["restitution"]

    if height < 0:
        height = 0.0  # clamp back to the floor
        velocity = -restitution * velocity  # reverse and scale velocity

    return np.array([height, velocity])


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * velocity**2  # mass = 1 (unit mass, cancels out)
    potential_energy = gravity * height
    return kinetic_energy, potential_energy