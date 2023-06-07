import copy

import numpy as np
import pandas as pd


# [ts_std_10, ts_mean_10, ts_max_10, ts_mean_20, ts_max_20, ts_min_10, ts_min_20, ts_sum_10, ts_sum_5, ts_std_20,
#                delta_1, delta_3, delta_5, delay_1, delay_3, delay_5, delay_5, delta_10, signed_power_2, ts_sum_3, ts_mean_5, ts_max_5, ts_min_5]
def rolling_window(a, window, axis=0):
    """
    返回2D array的滑窗array的array
    """
    if axis == 0:
        shape = (a.shape[0] - window + 1, window, a.shape[-1])
        strides = (a.strides[0],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    elif axis == 1:
        shape = (a.shape[-1] - window + 1,) + (a.shape[0], window)
        strides = (a.strides[-1],) + a.strides
        a_rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a_rolling


def rolling_nanmean(A, window=None):
    ret = pd.DataFrame(A)
    factor_table = copy.deepcopy(ret)
    for col in ret.columns:
        current_data = copy.deepcopy(ret[col])
        current_data.dropna(inplace=True)
        current_factor = current_data.rolling(window).mean().values
        number = 0
        for index, data in enumerate(ret[col]):

            if ret[col][index] != ret[col][index]:
                factor_table[col][index] = np.nan
            else:
                factor_table[col][index] = current_factor[number]
                number += 1
    factor = factor_table.to_numpy(dtype=np.double)
    return factor


def rolling_max(A, window=None):
    # ret = np.full(A.shape, np.nan)
    # A_rolling = rolling_window(A, window=window, axis=0)
    # Atmp = np.stack(map(lambda x:np.max(x, axis=0), A_rolling))
    # ret[window-1:,:] = Atmp
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).max()
    factor = factor.to_numpy(dtype=np.double)
    return factor


def rolling_nanstd(A, window=None):
    ret = pd.DataFrame(A)
    factor_table = copy.deepcopy(ret)
    for col in ret.columns:
        current_data = copy.deepcopy(ret[col])
        current_data.dropna(inplace=True)
        current_factor = current_data.rolling(window).std().values
        number = 0
        for index, data in enumerate(ret[col]):

            if ret[col][index] != ret[col][index]:
                factor_table[col][index] = np.nan
            else:
                factor_table[col][index] = current_factor[number]
                number += 1
    factor = factor_table.to_numpy(dtype=np.double)
    return factor


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def _ts_std(x1, t):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_nanstd(x1, t)


def _ts_mean(x1, t):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_nanmean(x1, t)


def _ts_max(x1, t):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_max(x1, t)

def _ts_std_10(x1):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_nanstd(x1, 10)


def _ts_mean_10(x1):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_nanmean(x1,10)


def _ts_max_10(x1):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_max(x1,10)


from functions import _Function

dynamic_ts_std = _Function(function=_ts_std, name='dynamic_ts_std', arity=1, isRandom=(True,(1,100)))
dynamic_ts_mean = _Function(function=_ts_mean, name='dynamic_ts_mean', arity=1, isRandom=(True,(1,100)))
dynamic_ts_max = _Function(function=_ts_max, name='dynamic_ts_max', arity=1, isRandom=(True,(1,100)))

ts_std_10 = _Function(function=_ts_std_10, name='ts_std_10', arity=1)
ts_mean_10 = _Function(function=_ts_mean_10, name='ts_mean_10', arity=1)
ts_max_10 = _Function(function=_ts_max_10, name='ts_max_10', arity=1)


_extra_function_map = {
    "dynamic_ts_std":dynamic_ts_std,
    "dynamic_ts_mean":dynamic_ts_mean,
    "dynamic_ts_max":dynamic_ts_max,

    "ts_std_10":ts_std_10,
    "ts_mean_10":ts_mean_10,
    "ts_max_10":ts_max_10,
}
