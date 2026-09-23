def rk4(dynamics, t, state, params, timestep):
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + timestep / 2, state + timestep / 2 * k1, params)
    k3 = dynamics(t + timestep / 2, state + timestep / 2 * k2, params)
    k4 = dynamics(t + timestep, state + timestep * k3, params)
    return state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)