import numpy as np
from scipy.integrate import solve_ivp

def lotka_volterra(t, z, alpha=1.5, beta=1.0, delta=3.0, gamma=1.0):
    """
    Lotka-Volterra predator-prey system.
    
    Parameters:
        t (float): Time (ignored in autonomous systems).
        z (array-like): State vector [prey, predator].
        alpha, beta, delta, gamma (float): System parameters.
    
    Returns:
        dzdt (list): Derivatives [dx/dt, dy/dt].
    """
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def simulate(
    t_max=20.0,
    dt=0.01,
    z0=(1.0, 1.0),
    solver_method="RK45",
    alpha=1.5,
    beta=1.0,
    delta=3.0,
    gamma=1.0,
    rtol=1e-9,
    atol=1e-9,
):
    """
    Simulate the Lotka-Volterra system.

    Parameters:
        t_max (float): Simulation duration.
        dt (float): Time step.
        z0 (tuple): Initial conditions (x0, y0).
        solver_method (str): Solver method (e.g., 'RK45', 'DOP853').
        alpha, beta, delta, gamma (float): System parameters.
        rtol, atol (float): Solver tolerances.

    Returns:
        t_eval (np.ndarray): Time points.
        trajectory (np.ndarray): State trajectory (N x 2).
    """
    t_eval = np.arange(0.0, t_max, dt)

    sol = solve_ivp(
        lotka_volterra,
        t_span=(0.0, t_max),
        y0=z0,
        t_eval=t_eval,
        method=solver_method,
        args=(alpha, beta, delta, gamma),
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return t_eval, sol.y.T