"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):
    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity,isRandom=(False,(1,100))):
        self.function = function
        self.name = name
        self.arity = arity

        self.isRandom = isRandom[0]

        self.RandRange = isRandom[1]
        if (not isinstance(self.RandRange,tuple)) or (not isinstance(self.RandRange[0],int)) or (not isinstance(self.RandRange[1],int)) or len(self.RandRange)!=2:
            raise TypeError("RandRange 格式错误，应该是类似(1,100)的tuple")
        self.baseConst = -1



    def __call__(self, *args):
        if self.isRandom and self.baseConst>0:
            if len(args)>1 and isinstance(args[-1],int):
                return self.function(*args)
            return self.function(*args,self.baseConst)
        else:
            return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _add(X: pd.DataFrame):
    return X.iloc[:, 0] + X.iloc[:, 1]


def _sub(X: pd.DataFrame):
    return X.iloc[:, 0] - X.iloc[:, 1]


def _mul(X: pd.DataFrame):
    return X.iloc[:, 0] * X.iloc[:, 1]


def _mul(X: pd.DataFrame):
    return X.iloc[:, 0] * X.iloc[:, 1]


def _div(X: pd.DataFrame):
    return X.iloc[:, 0] / X.iloc[:, 1]


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def rolling_window(a, window):
    a = np.asarray(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def ts_std_10(x1):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nanstd(rolling_window(x1, 10), axis=x1.ndim)


add2 = _Function(function=np.add, name='common_add', arity=2)
sub2 = _Function(function=np.subtract, name='common_sub', arity=2)
mul2 = _Function(function=np.multiply, name='common_mul', arity=2)
div2 = _Function(function=_protected_division, name='common_div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='common_sqrt', arity=1)
log1 = _Function(function=_protected_log, name='common_log', arity=1)
neg1 = _Function(function=np.negative, name='common_neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='common_inv', arity=1)
abs1 = _Function(function=np.abs, name='common_abs', arity=1)
max2 = _Function(function=np.maximum, name='common_max', arity=2)
min2 = _Function(function=np.minimum, name='common_min', arity=2)
sin1 = _Function(function=np.sin, name='common_sin', arity=1)
cos1 = _Function(function=np.cos, name='common_cos', arity=1)
tan1 = _Function(function=np.tan, name='common_tan', arity=1)
sig1 = _Function(function=_sigmoid, name='common_sig', arity=1)
# std_10 = _Function(function=ts_std_10, name='sig', arity=1)
# _function_map = {'add': add2,
#                  'sub': sub2,
#                  'mul': mul2,
#                  'div': div2,
#                  'sqrt': sqrt1,
#                  'log': log1,
#                  'abs': abs1,
#                  'neg': neg1,
#                  'inv': inv1,
#                  'max': max2,
#                  'min': min2,
#                  'sin': sin1,
#                  'cos': cos1,
#                  'tan': tan1}
_function_map = {
    'common_add': add2,
    'common_sub': sub2,
    'common_mul': mul2,
    'common_div': div2,
    'common_sqrt': sqrt1,
    'common_log': log1,
    'common_abs': abs1,
    'common_neg': neg1,
    'common_inv': inv1,
    'common_max': max2,
    'common_min': min2,
    'common_sin': sin1,
    'common_cos': cos1,
    'common_tan': tan1
}
