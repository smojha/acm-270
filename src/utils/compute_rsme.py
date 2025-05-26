import numpy as np

def compute_rmse_over_time(true_list, pred_list):
    """
    Compute RMSE[t] = sqrt( (1/M) * sum_i ||pred_i[t] - true_i[t]||^2 )
    across M trajectory pairs at each time step t.

    Parameters:
        true_list (List[np.ndarray]): List of true trajectories, shape (T, D) per entry
        pred_list (List[np.ndarray]): List of predicted trajectories, shape (T, D) per entry

    Returns:
        rmse_over_time (np.ndarray): Array of shape (T,), one RMSE value per time step
    """
    # Check consistency
    assert len(true_list) == len(pred_list), "Number of trajectories must match"
    M = len(true_list)
    T = true_list[0].shape[0]

    # Stack into (M, T, D)
    true_stack = np.stack(true_list, axis=0)
    pred_stack = np.stack(pred_list, axis=0)

    # Compute per-sample L2 error over time: shape (M, T)
    squared_l2_errors = np.sum((pred_stack - true_stack) ** 2, axis=2)

    # Mean over M and sqrt to get RMSE[t]
    rmse_over_time = np.sqrt(np.mean(squared_l2_errors, axis=0))

    return rmse_over_time