def explicit_euler(dynamics, t, state, params, timestep):
    return state + timestep * dynamics(t, state, params)