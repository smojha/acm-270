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

def simulate(t_max=20.0, dt=0.01, z0=(1.0, 1.0), **params):
    """
    Simulate the Lotka-Volterra system over time.
    
    Parameters:
        t_max (float): Maximum time to simulate.
        dt (float): Time step for evaluation.
        z0 (tuple): Initial condition (x0, y0).
        params (dict): Optional overrides for alpha, beta, delta, gamma.
    
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
        args=tuple(params.get(k, v) for k, v in zip(["alpha", "beta", "delta", "gamma"], [1.5, 1.0, 3.0, 1.0])),
        rtol=1e-9,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError("Integration failed.")

    return t_eval, sol.y.T  # shape: (N, 2)