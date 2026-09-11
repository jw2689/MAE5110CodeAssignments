import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from models import rimless_wheel as model
from integrators import rk4 as integrator

params = model.generate_params(num_spokes=8, slope_angle=0.2)
alpha = params["half_spoke_angle"]
gamma = params["slope_angle"]
post_impact_angle = gamma - alpha

def step_to_next_impact(t0, state0, params, coarse_timestep=2e-2, max_time=8.0):
    """Advance until the impact guard crosses zero, then bisect on the
    sub-step size to locate it precisely. Returns (t_impact, pre_impact_state,
    post_impact_state), or None if no impact happens within max_time."""
    t, state = t0, state0
    guard_prev = model.detect_impact(state, params)
    elapsed = 0.0

    while elapsed < max_time:
        next_state = integrator(model.compute_state_derivative, t, state, params, coarse_timestep)
        guard_next = model.detect_impact(next_state, params)

        if guard_prev < 0 and guard_next >= 0:
            low, high = 0.0, coarse_timestep
            for _ in range(50):
                mid = 0.5 * (low + high)
                candidate = integrator(model.compute_state_derivative, t, state, params, mid)
                guard_mid = model.detect_impact(candidate, params)
                if abs(guard_mid) < 1e-10:
                    break
                if np.sign(guard_mid) == np.sign(guard_prev):
                    low = mid
                else:
                    high = mid
            pre_impact_state = candidate
            post_impact_state = model.apply_impact_reset(pre_impact_state, params)
            return t + mid, pre_impact_state, post_impact_state

        t += coarse_timestep
        elapsed += coarse_timestep
        state, guard_prev = next_state, guard_next

    return None  # stalled: never reached the guard again


def simulate_one_period(state, params, dense_timestep=1e-3):
    """Simulate the smooth swing arc until the next impact. Returns the
    dense arc (for plotting) plus the pre- and post-impact states."""
    t = 0.0
    arc = [state.copy()]
    guard_prev = model.detect_impact(state, params)

    while True:
        next_state = integrator(model.compute_state_derivative, t, state, params, dense_timestep)
        guard_next = model.detect_impact(next_state, params)

        if guard_prev < 0 and guard_next >= 0:
            low, high = 0.0, dense_timestep
            for _ in range(50):
                mid = 0.5 * (low + high)
                candidate = integrator(model.compute_state_derivative, t, state, params, mid)
                guard_mid = model.detect_impact(candidate, params)
                if abs(guard_mid) < 1e-12:
                    break
                if np.sign(guard_mid) == np.sign(guard_prev):
                    low = mid
                else:
                    high = mid
            pre_impact_state = candidate
            arc.append(pre_impact_state.copy())
            break

        t += dense_timestep
        state = next_state
        guard_prev = guard_next
        arc.append(state.copy())

    post_impact_state = model.apply_impact_reset(pre_impact_state, params)
    return np.array(arc), pre_impact_state, post_impact_state


# Region of attraction

def classify_initial_condition(state, params, max_impacts=20, **kwargs):
    """'rolling' if it keeps impacting for max_impacts steps (converging to
    the limit cycle), 'stalled' if it ever fails to reach the guard."""
    t = 0.0
    for _ in range(max_impacts):
        result = step_to_next_impact(t, state, params, **kwargs)
        if result is None:
            return "stalled"
        t, _, state = result
    return "rolling"


def compute_roa_grid(params, theta_range, theta_dot_range, grid_size=40, **kwargs):
    theta_vals = np.linspace(*theta_range, grid_size)
    theta_dot_vals = np.linspace(*theta_dot_range, grid_size)
    classification = np.zeros((grid_size, grid_size))

    for i, theta in enumerate(theta_vals):
        for j, theta_dot in enumerate(theta_dot_vals):
            outcome = classify_initial_condition(np.array([theta, theta_dot]), params, **kwargs)
            classification[j, i] = 1.0 if outcome == "rolling" else 0.0

    return theta_vals, theta_dot_vals, classification


def compute_roa_fraction(params, theta_dot_range=(-4.0, 4.0), grid_size=15,
                          coarse_timestep=3e-2, max_time=4.0, max_impacts=15):
    """Fraction of a (theta, theta_dot) grid that converges to rolling.

    Kept fast on purpose: stalled points are the expensive case (they run
    all the way to max_time before giving up), so max_time is kept short --
    4s is already generous, since a trajectory that hasn't impacted again by
    then is essentially certain to be stalled. grid_size and max_impacts are
    also reduced relative to the main RoA plot, since this only needs to
    produce a smooth trend across many parameter values, not a
    publication-quality boundary."""
    a, g = params["half_spoke_angle"], params["slope_angle"]
    theta_range = (g - a, a + g)
    _, _, classification = compute_roa_grid(
        params, theta_range, theta_dot_range, grid_size=grid_size,
        coarse_timestep=coarse_timestep, max_time=max_time, max_impacts=max_impacts,
    )
    return classification.mean()


# Poincare return map, fixed point, Floquet multiplier

def next_post_impact_velocity(v0, params, **kwargs):
    initial_state = np.array([params["slope_angle"] - params["half_spoke_angle"], v0])
    result = step_to_next_impact(0.0, initial_state, params, **kwargs)
    return None if result is None else result[2][1]


def compute_return_map(params, velocity_range, num_points=60, **kwargs):
    v0_vals = np.linspace(*velocity_range, num_points)
    v1_vals = np.array([
        (next_post_impact_velocity(v0, params, **kwargs) or np.nan)
        for v0 in v0_vals
    ])
    return v0_vals, v1_vals


def find_fixed_point(params, v_min=0.05, v_max=4.0, num_scan=60, tol=1e-9, **kwargs):
    v_vals = np.linspace(v_min, v_max, num_scan)
    valid_v, residuals = [], []
    for v in v_vals:
        v1 = next_post_impact_velocity(v, params, **kwargs)
        if v1 is not None:
            valid_v.append(v)
            residuals.append(v1 - v)

    bracket = None
    for i in range(len(valid_v) - 1):
        if np.sign(residuals[i]) != np.sign(residuals[i + 1]):
            bracket = (valid_v[i], valid_v[i + 1])
            break
    if bracket is None:
        raise ValueError("No rolling fixed point found in this velocity range.")

    low, high = bracket
    for _ in range(100):
        mid = 0.5 * (low + high)
        residual_mid = next_post_impact_velocity(mid, params, **kwargs) - mid
        if abs(residual_mid) < tol:
            return mid
        residual_low = next_post_impact_velocity(low, params, **kwargs) - low
        low, high = (mid, high) if np.sign(residual_mid) == np.sign(residual_low) else (low, mid)
    return 0.5 * (low + high)


def estimate_floquet_multiplier(params, v_star, eps=1e-4, **kwargs):
    """Perturb the post-impact velocity on both sides of the fixed point and
    estimate the local slope of R(v) there via a central difference:
        dR/dv |_{v*}  ~=  (R(v*+eps) - R(v*-eps)) / (2*eps)
    This slope IS the Floquet multiplier: |multiplier|<1 means perturbations
    shrink step to step (stable), >1 means they grow (unstable)."""
    v_plus = next_post_impact_velocity(v_star + eps, params, **kwargs)
    v_minus = next_post_impact_velocity(v_star - eps, params, **kwargs)
    return (v_plus - v_minus) / (2 * eps)


def sweep_parameter(param_values, make_params):
    """Cheap sweep: fixed point + Floquet multiplier only (bisection-based,
    no grid), safe to run at full resolution."""
    fixed_points, multipliers = [], []
    for value in param_values:
        p = make_params(value)
        try:
            v_star = find_fixed_point(p)
            multiplier = estimate_floquet_multiplier(p, v_star)
        except ValueError:
            v_star, multiplier = np.nan, np.nan
        fixed_points.append(v_star)
        multipliers.append(multiplier)
    return np.array(fixed_points), np.array(multipliers)


def sweep_roa_fraction(param_values, make_params, **kwargs):
    """Expensive sweep: RoA coverage. Kept at a coarser parameter resolution
    than sweep_parameter, since every point here costs a full grid."""
    return np.array([compute_roa_fraction(make_params(value), **kwargs) for value in param_values])


# Fixed point, needed by several plots below
v_star = find_fixed_point(params)
true_arc, true_pre, true_post = simulate_one_period(
    np.array([post_impact_angle, v_star]), params
)


# ---------------------------------------------------------------------------
# Region of attraction: red/blue grid classification, with a legend
# (not a colorbar), plus the limit cycle overlaid on top.
# ---------------------------------------------------------------------------

theta_range = (gamma - alpha, alpha + gamma)
theta_dot_range = (-4.0, 4.0)

theta_vals, theta_dot_vals, classification = compute_roa_grid(
    params, theta_range, theta_dot_range, grid_size=40
)

roa_cmap = ListedColormap(["tab:blue", "tab:red"])

plt.figure(figsize=(7, 5))
plt.pcolormesh(theta_vals, theta_dot_vals, classification, shading="auto", cmap=roa_cmap)

#plotting the limit cycle on this
#plt.plot(true_arc[:, 0], true_arc[:, 1], color="black", linewidth=2.5, zorder=5,
#          label=f"Limit cycle (v*={v_star:.3f})")
#plt.plot([true_pre[0], true_post[0]], [true_pre[1], true_post[1]],
#          "k--", linewidth=1.5, zorder=5)

plt.xlabel("Stance angle θ (rad)")
plt.ylabel("Stance angular velocity θ̇ (rad/s)")
plt.title(f"Region of attraction (γ={gamma:.2f} rad, N={params['num_spokes']})")
legend_handles = [
    Patch(facecolor="tab:red", label="Stable (converges to limit cycle)"),
    Patch(facecolor="tab:blue", label="Unstable (stalled)"),
    #Line2D([0], [0], color="black", lw=2.5, label=f"Limit cycle (v*={v_star:.3f})"),
]
plt.legend(handles=legend_handles, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("roa.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# Simulate every grid point, trace its full path, color by convergence
# ---------------------------------------------------------------------------

def simulate_dense_path(initial_state, params, timestep=2e-3, sim_time=6.0):
    """Integrate continuously (fixed timestep, linear-interpolation reset)
    for a fixed time window, returning the dense (theta, theta_dot) path --
    used here purely for visualization, not precise event timing."""
    n_steps = int(sim_time / timestep) + 1
    state_traj = np.zeros((2, n_steps))
    state_traj[:, 0] = initial_state

    for step in range(n_steps - 1):
        t = step * timestep
        prev_state = state_traj[:, step]
        next_state = integrator(model.compute_state_derivative, t, prev_state, params, timestep)

        guard_prev = model.detect_impact(prev_state, params)
        guard_next = model.detect_impact(next_state, params)

        if guard_prev < 0 and guard_next >= 0:
            frac = guard_prev / (guard_prev - guard_next)
            crossing_state = prev_state + frac * (next_state - prev_state)
            next_state = model.apply_impact_reset(crossing_state, params)

        state_traj[:, step + 1] = next_state

    return state_traj

'''
grid_size = 8  # kept modest since every point draws a full traced path
theta_starts = np.linspace(*theta_range, grid_size)
theta_dot_starts = np.linspace(*theta_dot_range, grid_size)

n_arrows_per_path = 4
look_ahead = 60  # steps between an arrow's tail and head, for visibility

plt.figure(figsize=(8, 7))

for theta0 in theta_starts:
    for theta_dot0 in theta_dot_starts:
        initial_state = np.array([theta0, theta_dot0])
        outcome = classify_initial_condition(initial_state, params)
        color = "tab:red" if outcome == "rolling" else "tab:blue"

        path = simulate_dense_path(initial_state, params)
        plt.plot(path[0, :], path[1, :], color=color, alpha=0.5, linewidth=0.8)

        path_len = path.shape[1]
        arrow_indices = np.linspace(0, path_len - look_ahead - 1, n_arrows_per_path, dtype=int)
        for idx in arrow_indices:
            x0, y0 = path[0, idx], path[1, idx]
            x1, y1 = path[0, idx + look_ahead], path[1, idx + look_ahead]
            plt.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", color=color, alpha=0.8, lw=1.2))

        plt.plot(theta0, theta_dot0, "o", color=color, markersize=3, zorder=3)

# overlay the limit cycle itself, bold and on top
plt.plot(true_arc[:, 0], true_arc[:, 1], color="black", linewidth=2.5, zorder=5,
          label=f"Limit cycle (v*={v_star:.3f})")
plt.plot([true_pre[0], true_post[0]], [true_pre[1], true_post[1]],
          "k--", linewidth=1.5, zorder=5)

plt.xlim(theta_range[0] - 1.1, theta_range[1] + 0.2)
plt.ylim(*theta_dot_range)
plt.xlabel("Stance angle θ (rad)")
plt.ylabel("Stance angular velocity θ̇ (rad/s)")
plt.title(f"Simulated trajectories from every grid point (γ={gamma:.2f} rad, N={params['num_spokes']})")

legend_handles = [
    Line2D([0], [0], color="tab:red", lw=2, label="Converges to limit cycle"),
    Line2D([0], [0], color="tab:blue", lw=2, label="Does not converge"),
    Line2D([0], [0], color="black", lw=2.5, label="Limit cycle"),
]
plt.legend(handles=legend_handles, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("roa_traced_paths.png", dpi=150)
plt.show()'''

# 1D Poincare return map, fixed point clearly labelled

v0_vals, v1_vals = compute_return_map(params, velocity_range=(0.1, 3.0))

plt.figure(figsize=(6, 5))
plt.plot(v0_vals, v1_vals, label="Return map R(v)")
plt.plot(v0_vals, v0_vals, "k--", label="Identity line")
plt.plot(v_star, v_star, "ro", markersize=10, zorder=5, label=f"Fixed point v*={v_star:.4f}")
plt.annotate(f"v* = {v_star:.4f} rad/s", xy=(v_star, v_star),
             xytext=(v_star + 0.4, v_star - 0.5),
             arrowprops=dict(arrowstyle="->", color="black"))
plt.xlabel("Post-impact velocity, step n (rad/s)")
plt.ylabel("Post-impact velocity, step n+1 (rad/s)")
plt.title(f"Poincare return map (γ={gamma:.2f} rad, N={params['num_spokes']})")
plt.legend()
plt.tight_layout()
plt.savefig("return_map.png", dpi=150)
plt.show()


# Floquet multiplier at baseline parameters

floquet_multiplier = estimate_floquet_multiplier(params, v_star)
predicted = np.cos(2 * alpha) ** 2  # closed-form check, derived by hand

print(f"Fixed point v* = {v_star:.4f} rad/s")
print(f"Floquet multiplier = {floquet_multiplier:.4f}  (predicted cos^2(2*alpha) = {predicted:.4f})")
print(f"Stable: |multiplier| < 1 -> {abs(floquet_multiplier) < 1}")


# Sweep slope angle gamma, and sweep spoke count N

slope_angles = np.linspace(0.05, 0.5, 20)
v_star_vs_gamma, multiplier_vs_gamma = sweep_parameter(
    slope_angles, lambda g: model.generate_params(num_spokes=8, slope_angle=g)
)

gamma_subset = np.linspace(0.05, 0.5, 10)  # coarser -- this is the expensive part
roa_fraction_vs_gamma = sweep_roa_fraction(
    gamma_subset, lambda g: model.generate_params(num_spokes=8, slope_angle=g)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(gamma_subset, roa_fraction_vs_gamma * 100, "o-", color="tab:red")
axes[0].set_xlabel("Slope angle γ (rad)")
axes[0].set_ylabel("% of grid converging (RoA coverage)")
axes[0].set_ylim(0, 100)
axes[0].set_title("Region of attraction coverage vs. slope (N=8)")
no_gait = roa_fraction_vs_gamma == 0
#if no_gait.any():
##    axes[0].annotate("no rolling gait exists here", xy=(gamma_subset[no_gait][0], 2),
 #                     xytext=(gamma_subset[no_gait][0] + 0.05, 20),
#                      arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

axes[1].plot(slope_angles, multiplier_vs_gamma, "o-")
axes[1].axhline(1, color="gray", linestyle="--")
axes[1].axhline(-1, color="gray", linestyle="--")
axes[1].set_xlabel("Slope angle γ (rad)")
axes[1].set_ylabel("Floquet multiplier")
axes[1].set_title("Stability vs. slope (N=8)")
plt.tight_layout()
plt.savefig("sweep_gamma.png", dpi=150)
plt.show()

print("\ngamma sweep -- fixed-point speed v* (for reference, not plotted above):")
for g, v in zip(slope_angles, v_star_vs_gamma):
    print(f"  gamma={g:.3f}  v*={v:.4f}")

spoke_counts = np.arange(6, 13)
v_star_vs_N, multiplier_vs_N = sweep_parameter(
    spoke_counts, lambda N: model.generate_params(num_spokes=N, slope_angle=0.2)
)

roa_fraction_vs_N = sweep_roa_fraction(
    spoke_counts, lambda N: model.generate_params(num_spokes=N, slope_angle=0.2)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(spoke_counts, roa_fraction_vs_N * 100, "o-", color="tab:red")
axes[0].set_xlabel("Number of spokes N")
axes[0].set_ylabel("% of grid converging (RoA coverage)")
axes[0].set_ylim(0, 100)
axes[0].set_title("Region of attraction coverage vs. spoke count (γ=0.2 rad)")

axes[1].plot(spoke_counts, multiplier_vs_N, "o-")
axes[1].axhline(1, color="gray", linestyle="--")
axes[1].set_xlabel("Number of spokes N")
axes[1].set_ylabel("Floquet multiplier")
axes[1].set_title("Stability vs. spoke count (γ=0.2 rad)")
plt.tight_layout()
plt.savefig("sweep_num_spokes.png", dpi=150)
plt.show()

print("\nN sweep -- fixed-point speed v* (for reference, not plotted above):")
for N, v in zip(spoke_counts, v_star_vs_N):
    print(f"  N={N:2d}  v*={v:.4f}")