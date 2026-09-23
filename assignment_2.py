from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator

params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
    "kp": 9.0, #trial and error
    "kd": 6.0, #trial and error for smooth but not too damped
}


def compute_ankle_torque(state, params):
    theta, theta_dot = state
    g = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    kp = params["kp"]
    kd = params["kd"]

    virtual_accel = -kp * theta - kd * theta_dot
    torque = mass * length**2 * (virtual_accel - (g / length) * np.sin(theta))

    return np.clip(torque, -0.1 * mass * g * length, 0.05 * mass * g * length)


def in_roa(theta0, theta_dot0, params, sim_time=5.0, dt=1e-3, tol=1e-2):
    """Note: works on a local copy of params, not the caller's dict --
    otherwise every call would mutate the shared params object's
    ankle_torque in place, which is fragile for the grid search calling
    this in a loop (per review comment)."""
    local_params = dict(params)
    state = np.array([theta0, theta_dot0])
    for _ in range(int(sim_time / dt)):
        local_params["ankle_torque"] = compute_ankle_torque(state, local_params)
        state = integrator(model.dynamics, 0, state, local_params, dt)
    return np.linalg.norm(state) < tol


def section_guard(previous_state, next_state, params):
    """Poincare section: theta = 0 (mid-stance, upright crossing).

    Unlike touchdown (theta_TD = alpha + gamma, which moves whenever alpha
    changes), theta = 0 is a fixed constant regardless of the chosen angle
    of attack, and is crossed transversally (theta_dot != 0) during normal
    forward walking -- so it works as a section here where touchdown no
    longer does.
    """
    theta_prev = previous_state[0]
    theta_next = next_state[0]
    return theta_prev < 0 <= theta_next


def section_crossing_theta_dot(previous_state, next_state):
    """Linearly interpolate theta_dot at the theta=0 crossing, giving the
    single Poincare-map state theta_dot_k."""
    theta_prev, theta_dot_prev = previous_state
    theta_next, theta_dot_next = next_state
    frac = -theta_prev / (theta_next - theta_prev)
    return theta_dot_prev + frac * (theta_dot_next - theta_dot_prev)


def simulate_one_step(theta_dot_k, alpha, params, roa_lookup, dt=1e-3, max_time=5.0):
    """Simulate one discrete footstep of the step-to-step dynamics.

    Starts at the Poincare section (theta=0, theta_dot_k), swings forward
    passively (ankle_torque=0 -- this is the discrete footstep controller;
    the continuous balance controller only takes over once inside its RoA)
    under the chosen angle of attack, hits touchdown, applies the impact
    map, then either:
      - the post-impact state is inside the standing RoA -> done in 1 step
      - otherwise, keep swinging passively to the next theta=0 crossing,
        returning theta_dot_{k+1} for the return map.

    roa_lookup: a callable(state) -> bool, e.g. a closure over in_roa_guard
    and your fitted roa_mask/theta_grid/theta_dot_grid.

    Returns (post_impact_state, next_theta_dot_or_None, reached_roa).
    """
    step_params = dict(params)
    step_params["angle_of_attack"] = alpha
    step_params["ankle_torque"] = 0.0

    n_steps = int(max_time / dt)
    state = np.array([0.0, theta_dot_k])

    # Phase 1: swing forward to touchdown.
    post_impact_state = None
    for _ in range(n_steps):
        next_state = integrator(model.dynamics, 0, state, step_params, dt)
        if model.event_guard(state, next_state, step_params):
            alpha_g = step_params["angle_of_attack"]
            gamma = step_params["incline"]
            guard_prev = state[0] - (alpha_g + gamma)
            guard_next = next_state[0] - (alpha_g + gamma)
            frac = guard_prev / (guard_prev - guard_next)
            crossing_state = state + frac * (next_state - state)
            post_impact_state = model.event_dynamics(crossing_state, step_params)
            break
        state = next_state
    if post_impact_state is None:
        raise RuntimeError(
            "Touchdown not reached within max_time "
            f"(theta_dot_k={theta_dot_k:.3f}, alpha={alpha:.3f})."
        )

    if roa_lookup(post_impact_state):
        return post_impact_state, None, True

    # Phase 2: continue passively toward the next theta=0 crossing, checking
    # the RoA at every timestep (not just once at the post-impact instant) --
    # the state may pass through the RoA partway through this swing.
    state = post_impact_state
    for _ in range(n_steps):
        next_state = integrator(model.dynamics, 0, state, step_params, dt)
        if roa_lookup(next_state):
            return post_impact_state, None, True
        if section_guard(state, next_state, step_params):
            next_theta_dot = section_crossing_theta_dot(state, next_state)
            return post_impact_state, next_theta_dot, False
        state = next_state

    raise RuntimeError(
        "Poincare section not reached within max_time after impact "
        f"(theta_dot_k={theta_dot_k:.3f}, alpha={alpha:.3f})."
    )


def build_lookup_table(M, K, params, roa_lookup, alpha_bounds=(np.pi / 8, np.pi / 7)):
    """Build the (state, action) table and steps-to-standstill map at a given
    grid resolution (M theta_dot_k points, K alpha points). Returns a dict
    with everything needed to use or evaluate this table."""
    froude_2_theta_dot = np.sqrt(2 * params["gravity"] / params["length"])
    theta_dot_states = np.linspace(0.0, froude_2_theta_dot, M)
    alpha_actions = np.linspace(alpha_bounds[0], alpha_bounds[1], K)

    next_theta_dot_table = np.full((M, K), np.nan)
    reaches_roa_table = np.zeros((M, K), dtype=bool)

    for i, td in enumerate(theta_dot_states):
        for k, alpha in enumerate(alpha_actions):
            try:
                _, next_td, reached = simulate_one_step(td, alpha, params, roa_lookup)
            except RuntimeError:
                continue
            reaches_roa_table[i, k] = reached
            if not reached:
                next_theta_dot_table[i, k] = next_td

    steps_to_standstill = np.full(M, np.inf)
    best_action = np.full(M, -1, dtype=int)

    for i in range(M):
        if np.any(reaches_roa_table[i, :]):
            steps_to_standstill[i] = 1
            best_action[i] = int(np.argmax(reaches_roa_table[i, :]))

    changed = True
    while changed:
        changed = False
        for i in range(M):
            if np.isfinite(steps_to_standstill[i]):
                continue
            for k in range(K):
                if reaches_roa_table[i, k] or np.isnan(next_theta_dot_table[i, k]):
                    continue
                j = int(np.argmin(np.abs(theta_dot_states - next_theta_dot_table[i, k])))
                if np.isfinite(steps_to_standstill[j]):
                    candidate = steps_to_standstill[j] + 1
                    if candidate < steps_to_standstill[i]:
                        steps_to_standstill[i] = candidate
                        best_action[i] = k
                        changed = True

    return {
        "M": M, "K": K,
        "theta_dot_states": theta_dot_states,
        "alpha_actions": alpha_actions,
        "reaches_roa_table": reaches_roa_table,
        "next_theta_dot_table": next_theta_dot_table,
        "steps_to_standstill": steps_to_standstill,
        "best_action": best_action,
    }


def simulate_policy(theta_dot0, table, params, roa_lookup, max_steps=15):
    """Actually simulate (real continuous dynamics, not table lookup for the
    outcome) starting from theta_dot0, following the given table's policy
    (nearest-grid-point theta_dot_k -> best_action -> alpha) at each step.
    Returns the number of real footsteps to reach the RoA, or None if it
    doesn't reach it within max_steps (either the table has no policy at
    some point, or the walker takes longer than max_steps)."""
    theta_dot_states = table["theta_dot_states"]
    best_action = table["best_action"]
    alpha_actions = table["alpha_actions"]

    td = theta_dot0
    for n in range(1, max_steps + 1):
        idx = int(np.argmin(np.abs(theta_dot_states - td)))
        if best_action[idx] < 0:
            return None
        alpha = alpha_actions[best_action[idx]]
        try:
            _, next_td, reached = simulate_one_step(td, alpha, params, roa_lookup)
        except RuntimeError:
            return None
        if reached:
            return n
        td = next_td
    return None


def in_roa_guard(state, theta_grid, theta_dot_grid, roa_mask):
    """Nearest-grid-point lookup, but only within the grid's actual range --
    a state outside [theta_grid[0], theta_grid[-1]] x [theta_dot_grid[0],
    theta_dot_grid[-1]] is definitely not in the RoA; argmin alone would
    silently snap it to the nearest edge cell and could return a false
    positive."""
    theta, theta_dot = state
    if not (theta_grid[0] <= theta <= theta_grid[-1]):
        return False
    if not (theta_dot_grid[0] <= theta_dot <= theta_dot_grid[-1]):
        return False
    i = np.argmin(np.abs(theta_grid - theta))
    j = np.argmin(np.abs(theta_dot_grid - theta_dot))
    return roa_mask[i, j]



def simulate_policy_trajectory(theta_dot0, alpha_sequence_fn, params, roa_lookup,
                                 dt=1e-3, max_time_per_phase=5.0, max_steps=15):
    """Simulate the REAL continuous trajectory (every intermediate state, not
    just per-step endpoints) starting at the section (theta=0, theta_dot0),
    following a per-step policy alpha_sequence_fn(theta_dot_k) -> alpha,
    until the state enters the RoA or max_steps footsteps have elapsed.

    Returns (t_all, state_all, footstrike_times, n_steps, reached_roa).
    """
    n_phase_steps = int(max_time_per_phase / dt)
    t_all = [0.0]
    state_all = [np.array([0.0, theta_dot0])]
    footstrike_times = []
    t_clock = 0.0
    td = theta_dot0

    for step_num in range(1, max_steps + 1):
        alpha = alpha_sequence_fn(td)
        step_params = dict(params)
        step_params["angle_of_attack"] = alpha
        step_params["ankle_torque"] = 0.0

        state = state_all[-1]
        post_impact_state = None
        # Phase 1: swing to touchdown.
        for _ in range(n_phase_steps):
            next_state = integrator(model.dynamics, 0, state, step_params, dt)
            t_clock += dt
            t_all.append(t_clock)
            state_all.append(next_state)
            if model.event_guard(state, next_state, step_params):
                alpha_g = step_params["angle_of_attack"]
                gamma = step_params["incline"]
                gp = state[0] - (alpha_g + gamma)
                gn = next_state[0] - (alpha_g + gamma)
                frac = gp / (gp - gn)
                crossing = state + frac * (next_state - state)
                post_impact_state = model.event_dynamics(crossing, step_params)
                state_all[-1] = post_impact_state  # replace with post-impact state
                footstrike_times.append(t_clock)
                break
            state = next_state
        if post_impact_state is None:
            return np.array(t_all), np.array(state_all).T, footstrike_times, step_num - 1, False

        if roa_lookup(post_impact_state):
            return np.array(t_all), np.array(state_all).T, footstrike_times, step_num, True

        # Phase 2: swing back to the next section crossing.
        state = post_impact_state
        reached_mid_phase2 = False
        for _ in range(n_phase_steps):
            next_state = integrator(model.dynamics, 0, state, step_params, dt)
            t_clock += dt
            t_all.append(t_clock)
            state_all.append(next_state)
            if roa_lookup(next_state):
                return np.array(t_all), np.array(state_all).T, footstrike_times, step_num, True
            if section_guard(state, next_state, step_params):
                td = section_crossing_theta_dot(state, next_state)
                reached_mid_phase2 = True
                break
            state = next_state
        if not reached_mid_phase2:
            return np.array(t_all), np.array(state_all).T, footstrike_times, step_num, False

    return np.array(t_all), np.array(state_all).T, footstrike_times, max_steps, False


def build_max_steps_table(table, alpha_actions=None, step_cap=25):
    """Mirror of steps_to_standstill, but computing the MAXIMUM number of
    footsteps achievable before being forced into the RoA (i.e. the most a
    deliberately-delaying policy could sustain), rather than the minimum.

    CAUTION: unlike the minimum-steps version, this is NOT guaranteed to
    terminate on its own. Nearest-grid-point rounding can create a
    self-loop -- a state whose next_theta_dot_table entry rounds back to
    its OWN grid index -- which lets max_steps grow by +1 every pass
    forever (confirmed empirically: this happens at coarse resolutions,
    e.g. M=6). A self-loop is a grid-quantization artifact, not a real
    physical result, so step_cap bounds the search and any state that
    hits it is reported as having no well-defined maximum at this
    resolution, rather than hanging or returning a meaningless huge number.
    """
    M = table["M"]
    K = table["K"]
    theta_dot_states = table["theta_dot_states"]
    reaches_roa_table = table["reaches_roa_table"]
    next_theta_dot_table = table["next_theta_dot_table"]

    max_steps = np.full(M, -np.inf)
    best_action_max = np.full(M, -1, dtype=int)
    capped = np.zeros(M, dtype=bool)

    for i in range(M):
        if np.any(reaches_roa_table[i, :]):
            max_steps[i] = 1
            best_action_max[i] = int(np.argmax(reaches_roa_table[i, :]))

    changed = True
    n_iter = 0
    while changed and n_iter < step_cap:
        n_iter += 1
        changed = False
        for i in range(M):
            if capped[i]:
                continue
            for k in range(K):
                if reaches_roa_table[i, k]:
                    candidate = 1
                elif not np.isnan(next_theta_dot_table[i, k]):
                    j = int(np.argmin(np.abs(theta_dot_states - next_theta_dot_table[i, k])))
                    if not np.isfinite(max_steps[j]):
                        continue
                    candidate = max_steps[j] + 1
                else:
                    continue
                if candidate > max_steps[i]:
                    if candidate > step_cap:
                        capped[i] = True
                        max_steps[i] = np.inf  # "no well-defined maximum at this resolution"
                        changed = True
                        break
                    max_steps[i] = candidate
                    best_action_max[i] = k
                    changed = True

    return max_steps, best_action_max


# --- Compute the region of attraction ---
N = 30
theta_grid = np.linspace(-0.5, 0.5, N)
theta_dot_grid = np.linspace(-0.5, 1, N)
roa_mask = np.zeros((N, N), dtype=bool)
for i, th in enumerate(theta_grid):
    for j, thd in enumerate(theta_dot_grid):
        roa_mask[i, j] = in_roa(th, thd, params)

print(f"RoA contains {roa_mask.sum()} / {roa_mask.size} grid points")
print("in_roa(0, 0) =", in_roa(0.0, 0.0, params))

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

# --- RoA plot (report deliverable) ---
plt.figure()
plt.imshow(
    roa_mask.T,
    origin="lower",
    extent=[theta_grid[0], theta_grid[-1], theta_dot_grid[0], theta_dot_grid[-1]],
    aspect="auto",
    cmap="Greens",
)
plt.xlabel("theta (rad)")
plt.ylabel("theta_dot (rad/s)")
plt.title("Region of attraction")
plt.colorbar(label="In RoA")
plt.savefig(output / "roa.png", dpi=150)
plt.show()


def roa_lookup(state):
    return in_roa_guard(state, theta_grid, theta_dot_grid, roa_mask)


print("=" * 70)
print("CONTROL AS A LOOKUP TABLE")
print("=" * 70)

froude_2_theta_dot = np.sqrt(2 * params["gravity"] / params["length"])

# Dense set of probe velocities (not just a handful) spanning the whole
# reachable range. A small, arbitrarily-chosen set of test points can make
# a coarse grid look "good enough" purely by luck -- nearest-grid-point
# lookup makes the policy a step function, so a few probe points can
# happen to land in favorable cells even when the table is genuinely too
# coarse elsewhere. Testing against many points averages that luck out.
test_theta_dots = np.linspace(0.3, froude_2_theta_dot * 0.95, 15)

reference_table = build_lookup_table(35, 35, params, roa_lookup)
reference_steps = {
    td: simulate_policy(td, reference_table, params, roa_lookup) for td in test_theta_dots
}
print(f"Reference resolution: M={reference_table['M']}, K={reference_table['K']}")
print("Reference step counts:", {f"{td:.2f}": n for td, n in reference_steps.items()})

STEP_TOLERANCE = 1

def steps_match(a, b, tol=STEP_TOLERANCE):
    if a is None or b is None:
        return a == b
    return abs(a - b) <= tol

candidate_sizes = [(3, 3), (4, 4), (6, 6), (9, 9), (12, 12), (15, 15), (20, 20), (25, 25)]
chosen_MK = None
resolution_results = []

for (M_cand, K_cand) in candidate_sizes:
    cand_table = build_lookup_table(M_cand, K_cand, params, roa_lookup)
    cand_steps = {td: simulate_policy(td, cand_table, params, roa_lookup) for td in test_theta_dots}
    matches = all(steps_match(cand_steps[td], reference_steps[td]) for td in test_theta_dots)
    resolution_results.append((M_cand, K_cand, cand_steps, matches))
    print(f"M={M_cand:3d}, K={K_cand:2d}: steps={cand_steps} -> "
          f"{'MATCHES reference (within 1 step)' if matches else 'DOES NOT MATCH reference'}")
    if matches and chosen_MK is None:
        chosen_MK = (M_cand, K_cand)

if chosen_MK is None:
    print("WARNING: no candidate grid matched the reference -- using the "
          "reference resolution itself. Consider testing finer candidates.")
    chosen_MK = (reference_table["M"], reference_table["K"])
else:
    print(f"\nChosen (coarsest passing) resolution: M={chosen_MK[0]}, K={chosen_MK[1]}")
    idx = candidate_sizes.index(chosen_MK)
    if idx > 0:
        M_fail, K_fail, steps_fail, _ = resolution_results[idx - 1]
        print(f"For comparison, the next-coarser grid (M={M_fail}, K={K_fail}) "
              f"gives steps={steps_fail}, which does NOT match the reference "
              f"{ {f'{td:.2f}': n for td, n in reference_steps.items()} } -- "
              "confirming that resolution is NOT fine enough.")

n_probes = len(test_theta_dots)
sizes = [M_c for M_c, K_c, _, _ in resolution_results]
n_matching = [
    sum(steps_match(steps[td], reference_steps[td]) for td in test_theta_dots)
    for _, _, steps, _ in resolution_results
]
max_deviation = []
for _, _, steps, _ in resolution_results:
    devs = [
        abs(steps[td] - reference_steps[td])
        for td in test_theta_dots
        if steps[td] is not None and reference_steps[td] is not None
    ]
    max_deviation.append(max(devs) if devs else np.nan)

fig_res, (ax_count, ax_dev) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

ax_count.plot(sizes, n_matching, "o-", color="#23699b")
ax_count.axhline(n_probes, color="gray", linestyle=":", linewidth=1, label=f"All {n_probes} probes")
ax_count.axvline(chosen_MK[0], color="red", linestyle="--", label=f"Chosen M={chosen_MK[0]}")
ax_count.set_xlabel("Grid size M (state axis)")
ax_count.set_ylabel(f"Probe points matching reference (of {n_probes})")
ax_count.set_title("Grid resolution study:\nhow many of 15 probe velocities agree with reference")
ax_count.set_ylim(0, n_probes + 1)
ax_count.legend(fontsize=8)

ax_dev.plot(sizes, max_deviation, "o-", color="#df8a25")
ax_dev.axhline(STEP_TOLERANCE, color="gray", linestyle=":", linewidth=1, label=f"Tolerance ({STEP_TOLERANCE} step)")
ax_dev.axvline(chosen_MK[0], color="red", linestyle="--", label=f"Chosen M={chosen_MK[0]}")
ax_dev.set_xlabel("Grid size M (state axis)")
ax_dev.set_ylabel("Largest step-count deviation from reference")
ax_dev.set_title("Worst-case deviation across all 15 probes")
ax_dev.legend(fontsize=8)

fig_res.savefig(output / "grid_resolution_study.png", dpi=150)
plt.show()

M, K = chosen_MK
print(f"\nBuilding final lookup table at chosen resolution M={M}, K={K}...")

table = build_lookup_table(M, K, params, roa_lookup)
theta_dot_states = table["theta_dot_states"]
alpha_actions = table["alpha_actions"]
reaches_roa_table = table["reaches_roa_table"]
next_theta_dot_table = table["next_theta_dot_table"]
steps_to_standstill = table["steps_to_standstill"]
best_action = table["best_action"]

n_one_step = int(np.any(reaches_roa_table, axis=1).sum())
print(f"{n_one_step}/{M} initial theta_dot_k states have at least one alpha "
      "that reaches the RoA in a single footstep.")
print()
print("Steps to standstill by initial theta_dot_k (state -> best alpha -> steps):")
for i, (td, n) in enumerate(zip(theta_dot_states, steps_to_standstill)):
    if np.isfinite(n):
        alpha_used = alpha_actions[best_action[i]]
        print(f"  theta_dot_k = {td:5.3f} rad/s -> alpha = {alpha_used:.3f} rad -> {int(n)} step(s)")
    else:
        print(f"  theta_dot_k = {td:5.3f} rad/s -> UNREACHABLE within this grid/step cap")

fig_lut, (ax_table, ax_steps) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

im_lut = ax_table.imshow(
    reaches_roa_table.T,
    origin="lower",
    extent=[theta_dot_states[0], theta_dot_states[-1], alpha_actions[0], alpha_actions[-1]],
    aspect="auto",
    cmap="Purples",
)
ax_table.set_xlabel(r"State: $\dot\theta_k$ (rad/s)")
ax_table.set_ylabel(r"Control input: $\alpha$ (rad)")
ax_table.set_title("State-action lookup table\n(dark = reaches RoA in 1 footstep)")
fig_lut.colorbar(im_lut, ax=ax_table, label="Reaches RoA directly")

finite_mask = np.isfinite(steps_to_standstill)
ax_steps.step(
    theta_dot_states[finite_mask], steps_to_standstill[finite_mask],
    where="mid", color="#23699b", label="Steps to standstill",
)
ax_steps.scatter(theta_dot_states[finite_mask], steps_to_standstill[finite_mask], color="#23699b", zorder=5)
if np.any(~finite_mask):
    ax_steps.scatter(
        theta_dot_states[~finite_mask],
        np.zeros(np.sum(~finite_mask)),
        color="red",
        marker="x",
        label="Unreachable (within this grid)",
        zorder=5,
    )
ax_steps.set_xlabel(r"Initial state: $\dot\theta_k$ (rad/s)")
ax_steps.set_ylabel("Minimum footsteps to standstill")
ax_steps.set_title("Steps-to-standstill by initial condition")
ax_steps.set_yticks(range(0, int(np.nanmax(steps_to_standstill[finite_mask])) + 2))
ax_steps.legend(loc="upper left")
ax_steps.grid(alpha=0.3)

fig_lut.savefig(output / "steps_to_standstill.png", dpi=150)
plt.show()

print()
print("=" * 70)
print(">= 3-STEP TRAJECTORY AND MAXIMUM SUSTAINABLE STEPS")
print("=" * 70)

candidates_ge3 = [i for i in range(M) if np.isfinite(steps_to_standstill[i]) and steps_to_standstill[i] >= 3]
if not candidates_ge3:
    raise RuntimeError("No grid state reaches standstill in >= 3 steps -- widen the state grid.")

# theta_dot_0 = 3.5 gives a real optimal step count of 4 -- comfortably
# above the ">= 3 steps" requirement with margin, rather than landing
# right at the minimum.
example_theta_dot0 = 3.5
example_idx = int(np.argmin(np.abs(theta_dot_states - example_theta_dot0)))
print(f"Chosen initial condition: theta_dot_0 = {example_theta_dot0:.3f} rad/s "
      f"(optimal policy: {int(steps_to_standstill[example_idx])} steps)")

def optimal_alpha(td):
    idx = int(np.argmin(np.abs(theta_dot_states - td)))
    return alpha_actions[best_action[idx]]

t_opt, state_opt, footstrikes_opt, n_steps_opt, reached_opt = simulate_policy_trajectory(
    example_theta_dot0, optimal_alpha, params, roa_lookup
)
print(f"Simulated (optimal policy): {n_steps_opt} footsteps, reached RoA: {reached_opt}")

max_steps_table, best_action_max = build_max_steps_table(table)
max_steps_example = max_steps_table[example_idx]
print(f"Maximum sustainable steps from this same initial condition: "
      f"{int(max_steps_example) if np.isfinite(max_steps_example) else 'unbounded/undetermined'}")

def delaying_alpha(td):
    idx = int(np.argmin(np.abs(theta_dot_states - td)))
    if best_action_max[idx] < 0:
        return optimal_alpha(td)  # fall back if no delaying policy is known here
    return alpha_actions[best_action_max[idx]]

t_max, state_max, footstrikes_max, n_steps_max, reached_max = simulate_policy_trajectory(
    example_theta_dot0, delaying_alpha, params, roa_lookup
)
print(f"Simulated (maximally-delaying policy): {n_steps_max} footsteps, reached RoA: {reached_max}")

fig_ex, (ax_traj, ax_compare) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

ax_traj.plot(t_max, state_max[0, :], color="#df8a25", linewidth=5, alpha=0.6,
             label=f"Max-delaying policy ({n_steps_max} steps)", zorder=2)
for ft in footstrikes_max:
    ax_traj.axvline(ft, color="#df8a25", linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
ax_traj.plot(t_opt, state_opt[0, :], color="#23699b", linewidth=1.8, linestyle="--",
             label=f"Optimal policy ({n_steps_opt} steps)", zorder=3)
for ft in footstrikes_opt:
    ax_traj.axvline(ft, color="#23699b", linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)
ax_traj.axhline(0.0, color="gray", linewidth=0.5)
ax_traj.set_xlabel("Time (s)")
ax_traj.set_ylabel(r"$\theta$ (rad)")
ax_traj.set_title(fr"Trajectories from $\dot\theta_0$={example_theta_dot0:.3f} rad/s"
                   "\n(dotted lines = footstrikes)")
ax_traj.legend(fontsize=8)

ax_compare.bar(
    ["Optimal\n(fewest steps)", "Maximum sustainable\n(most steps)"],
    [n_steps_opt, n_steps_max],
    color=["#23699b", "#df8a25"],
)
ax_compare.set_ylabel("Number of footsteps to reach RoA")
ax_compare.set_title(fr"Steps to standstill from $\dot\theta_0$={example_theta_dot0:.3f} rad/s")
for idx_bar, val in enumerate([n_steps_opt, n_steps_max]):
    ax_compare.text(idx_bar, val + 0.05, str(val), ha="center")

fig_ex.savefig(output / "ge3_step_trajectory.png", dpi=150)
plt.show()

# --- Walking simulation, ankle controller gated by the RoA guard ---
initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 5.0
desired_number_of_steps = 5

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

section_crossings = []

for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]

    if in_roa_guard(state, theta_grid, theta_dot_grid, roa_mask):
        params["ankle_torque"] = compute_ankle_torque(state, params)
    else:
        params["ankle_torque"] = 0.0

    next_state = state + timestep * model.dynamics(t, state, params)

    if section_guard(state, next_state, params):
        theta_dot_k = section_crossing_theta_dot(state, next_state)
        section_crossings.append((t, theta_dot_k))

    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

print(f"Poincare section (theta=0) crossings recorded: {len(section_crossings)}")
for t_cross, theta_dot_k in section_crossings:
    print(f"  t = {t_cross:.4f} s, theta_dot_k = {theta_dot_k:.4f} rad/s")

crossing_theta_dots = [theta_dot_k for _, theta_dot_k in section_crossings]

fig_section, (ax_phase, ax_seq) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")

ax_phase.plot(state_traj[0, :], state_traj[1, :], color="#23699b", linewidth=1, label="Trajectory")
ax_phase.axvline(0.0, color="#df8a25", linestyle="--", linewidth=2, label=r"Section ($\theta=0$)")
ax_phase.scatter(
    [0.0] * len(crossing_theta_dots),
    crossing_theta_dots,
    color="#df8a25",
    zorder=5,
    label="Crossings",
)
ax_phase.set_xlabel(r"$\theta$ (rad)")
ax_phase.set_ylabel(r"$\dot\theta$ (rad/s)")
ax_phase.set_title("Phase portrait and Poincare section")
ax_phase.legend(loc="best", fontsize=8)

ax_seq.plot(range(len(crossing_theta_dots)), crossing_theta_dots, "o-", color="#df8a25")
ax_seq.set_xlabel("Crossing index k")
ax_seq.set_ylabel(r"$\dot\theta_k$ (rad/s)")
ax_seq.set_title("Poincare section sequence")

fig_section.savefig(output / "poincare_section.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()