from typing import Callable, Iterable, Union
from unicodedata import numeric
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided
import pandas as pd


def autocorr(a: Iterable[numeric], inf: int, sup: int, step: int) -> np.array:
    """
    Return the autocorrelation (from pandas) of a in range(inf, sup, step)

    :param np.array a: 1D array to perform autocorrelation onto
    :param int inf: lower border (included) to perform autocorrelation
    :param int sup: upper border (exluded) to perform autocorrelation
    :param int step: distance between steps to performa autocorrelation
    """
    sig = pd.Series(a.ravel())

    tmp = []
    for i in range(inf, sup, step):
        tmp.append(sig.autocorr(i))
    
    return tmp

def zscore(a: Iterable[numeric]) -> np.array:
    """
    return Zscored array
    :param np.array a: 1D array to zscore
    """

    return (a - np.mean(a))/np.std(a)

def normalize(a: Iterable[numeric], inf: numeric = 0., sup: numeric=1.)-> np.array:
    """
    normalize array between [inf; sup]
    :param np.array a: array
    :param num inf: lower bound
    :param num sup: upper bound

    """
    assert inf < sup, "lower bound >= higher bound"
    return ((a-np.min(a))/(np.max(a)-np.min(a))*(sup-inf))+inf

def sliding_window_view(x, window_shape, axis = None, *,
                        subok=False, writeable=False):

    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)

def reshape_bin(a: Iterable, binsize: int, func: Callable[[], Iterable] = None)-> np.array:
    """
    return array shaped in bin (convinient to apply func)
    if the bin size do not fit perfectly, discard some values.
    ex1: reshape_bin(np.arange(10), 3)
    // [[0,1,2],
        [3,4,5],
        [6,7,8]]

    ex2: reshape_bin(np.arange(10),3, np.sum)
    //[3,12,21]

    """
    a = a[:len(a)-len(a)%binsize]
    a = a.reshape((-1,binsize))
    if type(func)==type(None):
        return a
    else:
        return func(a, axis=1).ravel()

def metaplot(a:np.array, pos: Iterable[int], size: int = 1500, sense : Union[None, Iterable[int]] = None) -> np.array:
    """
    return signal windows around pos and mean.
    if sense is a "+" & "-" array, reverse the -
    """
    ret = []
    if type(sense)==type(None):
        for p in pos:
            ret.append(a[p-1500, p+1500])
        return ret, np.mean(ret, axis=0)
    
    else:
        for i,p in enumerate(pos):
            if sense[i]=="+":
                ret.append(a[p-1500, p+1500])
            else:
                ret.append(a[p-1500, p+1500][::-1])
        return ret, np.mean(ret, axis=0)

def consecutive(data: Iterable[numeric], stepsize: int = 1) -> np.array:
    return np.split(data, np.where(np.diff(data) <= stepsize)[0]+1)

def string2fasta(a: str, header: Union[numeric, str] = "", size: int = 120) -> str:

    """
    Transform string to fasta formatted string

    :param str a: string to transform
    :param header: The name of the sequence as required by fasta format (">header")
    :param int size: number of residue by line (usually fasta required 60 or 120 as a maximum). 
    """
    if header == "":
        f = header
    else:
        f = ">" + str(header) + "\n"

    while len(a)>0:
        try:
            f += a[:size]
            a = a[size:]
        except IndexError:
            f += a
            a = ""
        f += "\n"

    return f

