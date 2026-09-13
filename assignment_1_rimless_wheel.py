import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import rk4 as integrator

params = model.generate_params(num_spokes=8, slope_angle=0.2)

initial_state = np.array([0, 0.2])  # start near vertical, small forward spin

timestep = 1e-3
sim_time = 10.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

impact_times = []
impact_velocities = []  # (pre-impact, post-impact) angular velocity pairs

for step, t in enumerate(time_traj[:-1]):
    prev_state = state_traj[:, step]
    next_state = integrator(model.compute_state_derivative, t, prev_state, params, timestep)

    guard_prev = model.detect_impact(prev_state, params)
    guard_next = model.detect_impact(next_state, params)

    if guard_prev < 0 and guard_next >= 0:
        # linear interpolation to estimate the crossing state within this step
        frac = guard_prev / (guard_prev - guard_next)
        crossing_state = prev_state + frac * (next_state - prev_state)

        pre_impact_velocity = crossing_state[1]
        next_state = model.apply_impact_reset(crossing_state, params)
        post_impact_velocity = next_state[1]

        impact_times.append(t + frac * timestep)
        impact_velocities.append((pre_impact_velocity, post_impact_velocity))

    state_traj[:, step + 1] = next_state

# --- sanity checks ---

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
total_energy = kinetic_energy + potential_energy

plt.figure()
plt.plot(time_traj, state_traj[0, :])
plt.xlabel("Time (s)")
plt.ylabel("Stance angle θ (rad)")
plt.title("Rimless wheel: stance angle over time")
plt.savefig("theta_time.png", dpi=150)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(time_traj, total_energy)
for t_impact in impact_times:
    plt.axvline(t_impact, color="gray", linestyle="--", alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Specific Energy (per mass)")
plt.title("Rimless wheel: total energy (dashed lines = impacts)")
plt.tight_layout()
plt.savefig("total_energy.png", dpi=150)
plt.show()

#print(f"Number of impacts: {len(impact_times)}")
#if impact_velocities:
#    print("First few (pre, post) impact velocities:", impact_velocities[:5])