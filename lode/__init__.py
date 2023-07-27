import numba
import numpy as np
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:]),
        (numba.float64[:, :], numba.float64[:]),
    ],
    "(m, m) -> (m)",
)
def lode_coords(arr, out):
    I1 = np.trace(arr)
    arr2 = np.multiply(arr, arr)
    tr2 = np.trace(arr2)
    arr3 = np.multiply(arr2, arr)
    J2 = 0.5 * (np.trace(arr2) - (1/3) * (I1 ** 2))
    J3 = (1/3) * (np.trace(arr3) - tr2 * I1 + (2/9) * (I1 ** 3))

    out[0] = I1 / np.sqrt(3)                    # z
    out[1] = np.sqrt(2 * J2)                    # r
    # out[2] = (J3 / 2) * ((3 / J2) ** (3/2))     # cos(3 theta)
    out[2] = np.arccos(J3 / (J2 ** (3/2))) / 3  # theta