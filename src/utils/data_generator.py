import numpy as np
import os
import yaml

from systems.lotka_volterra import simulate as simulate_lv
from systems.lorenz import simulate as simulate_lorenz

SIMULATE_FUNCS = {
    "lotka_volterra": simulate_lv,
    "lorenz": simulate_lorenz
}

def generate_batch_from_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    simulate_fn = SIMULATE_FUNCS[config["simulate_fn"]]
    z0_range = config["z0_range"]
    z0_dim = config["z0_dim"]

    # Optional solver and model parameters
    solver_method = config.get("solver_method", "RK45")
    model_params = config.get("parameters", {})

    def z0_sampler():
        return np.random.uniform(z0_range[0], z0_range[1], size=z0_dim)

    # Wrap simulate_fn to inject method and parameters
    def wrapped_simulate_fn(t_max, dt, z0):
        return simulate_fn(
            t_max=t_max,
            dt=dt,
            z0=z0,
            solver_method=solver_method,
            **model_params
        )

    generate_batch(
        simulate_fn=wrapped_simulate_fn,
        z0_sampler=z0_sampler,
        n_trajectories=config.get("n_trajectories", 100),
        t_max=config.get("t_max", 20.0),
        dt=config.get("dt", 0.01),
        filename=config.get("filename", "output.npz"),
        save_dir=config.get("save_dir", "data"),
        metadata=config  # Save full config as metadata
    )

def generate_batch(simulate_fn, z0_sampler, n_trajectories, t_max, dt, filename, save_dir, metadata=None):
    os.makedirs(save_dir, exist_ok=True)
    all_trajectories = []

    for i in range(n_trajectories):
        z0 = z0_sampler()
        try:
            t, traj = simulate_fn(t_max=t_max, dt=dt, z0=z0)
            all_trajectories.append({
                "z0": z0,
                "t": t,
                "traj": traj
            })
        except Exception as e:
            print(f"[Warning] Trajectory {i} failed: {e}")

    filepath = os.path.join(save_dir, filename)
    
    if metadata is None:
        metadata = {}
    np.savez_compressed(
        filepath,
        trajectories=all_trajectories,
        config=np.string_(yaml.dump(metadata))
    )

    print(f"[!] Saved {len(all_trajectories)} trajectories + config to {filepath}")