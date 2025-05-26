import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, xyz, sigma, rho, beta):
    """Lorenz system of equations."""
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def simulate(
    t_max=20.0,
    dt=0.01,
    z0=(1.0, 1.0, 1.0),
    solver_method="RK45",
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3,
    rtol=1e-9,
    atol=1e-9,
):
    """
    Simulate the Lorenz system.

    Parameters:
        t_max (float): maximum simulation time.
        dt (float): time step for evaluation points.
        z0 (tuple): initial condition (x0, y0, z0).
        params (dict): optional parameters for sigma, rho, beta.

    Returns:
        t_eval (np.ndarray): time points.
        trajectory (np.ndarray): system trajectory, shape (N, 3).
    """
    t_eval = np.arange(0.0, t_max, dt)
    sol = solve_ivp(
        lorenz,
        t_span=(0.0, t_max),
        y0=z0,
        t_eval=t_eval,
        method=solver_method,
        args=(sigma, rho, beta),
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError("Integration failed.")

    return t_eval, sol.y.T