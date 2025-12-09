import numpy as np
import torch

from silvr.utils.eval_unc import compute_ause


def sparsification_plot_original(var_vec, err_vec, uncert_type="c"):
    # https://github.com/abdo-eldesokey/pncnn/blob/c6122e9c442eabeb0145b241121aeba0039eb5e7/utils/sparsification_plot.py#L10
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)

    # Sort the error
    # print('Sorting Error ...')
    err_vec_sorted, _ = torch.sort(err_vec)
    # print(' Done!')

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    rmse_err = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted[0 : int((1 - r) * n_valid_pixels)]
        rmse_err.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())

    # Normalize RMSE
    rmse_err = rmse_err / rmse_err[0]

    ###########################################

    # Sort by variance
    # print('Sorting Variance ...')
    if uncert_type == "c":
        var_vec = torch.sqrt(var_vec)
        _, var_vec_sorted_idxs = torch.sort(var_vec, descending=True)
    elif uncert_type == "v":
        # var_vec = torch.exp(var_vec)
        var_vec = torch.sqrt(var_vec)
        _, var_vec_sorted_idxs = torch.sort(var_vec, descending=False)
    # print(' Done!')

    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]

    rmse_err_by_var = []
    for i, r in enumerate(ratio_removed):
        mse_err_slice = err_vec_sorted_by_var[0 : int((1 - r) * n_valid_pixels)]
        rmse_err_by_var.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())

    # Normalize RMSE
    rmse_err_by_var = rmse_err_by_var / max(rmse_err_by_var)

    # plt.plot(ratio_removed, rmse_err, '--')
    # plt.plot(ratio_removed, rmse_err_by_var, '-r')
    # plt.show()
    return rmse_err, rmse_err_by_var


# test a predifined error and unc, and test if the function returns the same result
def test_ause():
    var_vec = torch.rand(100)
    err_vec = torch.rand(100)
    rmse_err, rmse_err_by_var = sparsification_plot_original(var_vec, err_vec, uncert_type="v")
    rmse_err_np, rmse_err_by_var_np, ause, _, __ = compute_ause(var_vec.numpy(), err_vec.numpy(), "mse", old_impl=True)
    print(rmse_err)
    print(rmse_err_by_var)
    print(ause)
    assert np.allclose(rmse_err, rmse_err_np)
    assert np.allclose(rmse_err_by_var, rmse_err_by_var_np)
