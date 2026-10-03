import numpy as np


def dynamics(t, state, params):
    """Continuous (single-stance) dynamics: identical to an inverted pendulum.

    Note: independent of slope angle gamma -- gravity is fixed in the world
    frame, and theta is measured from vertical, so gamma only affects the
    impact guard/reset, not the swing equation itself.
    """
    gravity = params["gravity"]
    spoke_length = params["spoke_length"]

    stance_angle = state[0]
    stance_angular_velocity = state[1]

    angular_acceleration = (gravity / spoke_length) * np.sin(stance_angle)

    return np.array([stance_angular_velocity, angular_acceleration])


def generate_params(num_spokes=8, slope_angle=0.2):
    return {
        "gravity": 9.81,  # gravity (m/s^2)
        "spoke_length": 1,  # spoke length l (m)
        "num_spokes": num_spokes,  # N
        "half_spoke_angle": np.pi / num_spokes,  # alpha = pi / N
        "slope_angle": slope_angle,  # gamma (radians), downhill incline
    }


def generate_initial_condition():
    """Just after an impact (theta = gamma - alpha for the defaults), rolling forward."""
    return np.array([0.2 - np.pi / 8, 1.0])


def detect_impact(state, params):
    """Guard function. Zero-crossing (negative -> positive) signals impact."""
    stance_angle = state[0]
    alpha = params["half_spoke_angle"]
    gamma = params["slope_angle"]
    return stance_angle - (alpha + gamma)


def apply_impact_reset(state, params):
    """Instantaneous plastic collision: switch stance leg, conserve angular
    momentum about the new contact point."""
    stance_angle = state[0]
    stance_angular_velocity = state[1]
    alpha = params["half_spoke_angle"]
    gamma = params["slope_angle"]

    new_stance_angle = 2 * gamma - stance_angle
    new_stance_angular_velocity = stance_angular_velocity * np.cos(2 * alpha)

    return np.array([new_stance_angle, new_stance_angular_velocity])


def calculate_energy(state, params):
    """Nondimensional-style energy (unit mass): conserved during single-stance
    flight, drops at each impact. Useful as a sanity check."""
    gravity = params["gravity"]
    length = params["spoke_length"]

    stance_angle = state[0]
    stance_angular_velocity = state[1]

    kinetic_energy = 0.5 * (length * stance_angular_velocity) ** 2
    potential_energy = gravity * length * np.cos(stance_angle)
    return kinetic_energy, potential_energy