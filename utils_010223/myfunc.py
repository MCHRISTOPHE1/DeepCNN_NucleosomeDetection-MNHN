from typing import Callable, Iterable, Union
from unicodedata import numeric

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

BASES = np.array(["A", "C", "G", "T"])
#DOCSTRING TO REVIEW
def loadnp(l:str):
        if l[-1] == "y":
            return np.load(l)
        elif l[-1] == "z":
            return np.load(l)["arr_0"]
        else:
            raise ValueError("file: '{}' is nor .npz nor .npy".format(l))

            
def autocorr(a: Iterable[numeric], inf: int, sup: int, step: int) -> np.array:
    
    """
    Return the autocorrelation (from pandas) of a in range(inf, sup, step).

    ### Parameters
    1. a : np.array
        - 1D array to perform autocorrelation onto
    2. inf : int
        lower border (included) to perform autocorrelation
    3. sup : int
        - upper border (exluded) to perform autocorrelation
    4. step : int
        - distance between steps to performa autocorrelation

    ### Returns
    - np.array:
        1D Array containing the autocorrelation for the given offsets
    """
    
    sig = pd.Series(a.ravel())

    tmp = []
    for i in range(inf, sup, step):
        tmp.append(sig.autocorr(i))
    
    return tmp

def zscore(a: Iterable[numeric]) -> np.array:
    """
    return Zscored array

    ### Parameters
    1. a: np.array
        - 1D array to zscore

    ### Returns
    - np.array
        Same array zscored
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

def metaplot_signal(a:np.array, pos: Iterable[int], size: int = 1500, sense : Union[None, Iterable[int]] = None) -> np.array:
    """
    return signal windows around pos and mean.
    if sense is a "+" & "-" array, reverse the -
    """
    a = a.ravel()

    ret = []
    if type(sense)==type(None):
        for p in pos:
            ret.append(a[p-size : p+size])
        return ret, np.mean(ret, axis=0)
    
    else:
        for i,p in enumerate(pos):
            if sense[i]=="+":
                ret.append(a[p-size : p+size])
            else:
                ret.append(a[p-size : p+size][::-1])
        return ret, np.mean(ret, axis=0)

def consecutive(data: Iterable[numeric], stepsize: int = 1) -> np.array:
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

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

def metaplot_V0(experimental: np.array, predicted: np.array, pos: Iterable, sense: Union[None, Iterable] = None,
                srange: int = 1500, mutasome: Union[None, np.array] = None, zscored:bool = True,
                display: bool = True, size:tuple = (12,12),
                 save: bool = False, path: str = ""):
    """
    Plot and or save two metaplots side by side (individual + mean)
    around the provided position and for the provided size around it.
    Individuals are sorted by their correlation between data.

    ### Parameters
    1. experimental: np.array
        - Experimental array (ground truth)

    2. predicted: np.array
        - Predicted array (must be aligned with experimental)

    3. pos: Iterable
        - Iterable of the positions to use

    4. sense: Union(None, Iterable)
        - Strand ["+", "-"]

    4. range: int
        - length of the signal to take around the position from either side

    5. mutasome: np.array
        -Mutasome array

    6. zscore: bool
        - Apply zscore on the signals

    7. display: bool
        - if True, display the signal (matplotlib used).
        This only work is your interactive mode is turned off.
    
    8. size: tuple
        - size of the figure (width, height)

    9. save: bool
        - if True save the signal at the indicated path

    10. path
        - path to save the figure

    ### Returns
    figure and axs
    """

    if zscored:
        experimental = zscore(experimental)
        predicted = zscore(predicted)

    if type(sense) == type(None):
        sense = ["+" for i in range(len(pos))]

    tmp1, tmp2, corr =  [], [], []
    for i,c in enumerate(pos):
        e, p = np.array(experimental[c-srange: c+srange+1]).ravel(), np.array(predicted[c-srange:c + srange +1 ]).ravel()
        if len(e) == 2*srange+1 and len(p) == 2*srange+1:
            if sense[i]=="+":
                tmp1.append(e)
                tmp2.append(p)
            else:
                tmp1.append(e[::-1])
                tmp2.append(p[::-1])
            corr.append(np.corrcoef(tmp1[-1], tmp2[-1])[0][1])


    idx = np.argsort(corr)[::-1]

    tmp1 = np.array(tmp1)[idx].reshape((-1,2*srange+1))
    m1   = np.mean(tmp1,axis=0)
    tmp2 = np.array(tmp2)[idx].reshape((-1,2*srange+1))
    m2   = np.mean(tmp2,axis=0)
    corr = np.array(corr)[idx].reshape(-1,1)

    #Graphical
    cmap = mpl.cm.YlOrRd
    mosaic = [["line1", ".", "line2"],
                ["line1","corr_legend", "line2"],
                ["plot1","corr", "plot2"]]

    fig, axs = plt.subplot_mosaic(mosaic, figsize = size, gridspec_kw={"width_ratios":[5,1,5], "height_ratios":[10,1,30], "hspace":0.0, "wspace":0.1})
    axs["line1"].plot(m1)
    axs["line1"].set_xticks([])
    axs["line1"].set_title("EXPERIMENTAL", fontsize=25)
    axs["line1"].set_ylabel("Nucleosome density")
    

    axs["line2"].plot(m2)
    axs["line2"].set_xticks([])
    axs["line2"].set_title("PREDICTION", fontsize=25)
    axs["line2"].set_yticks([])
   
    upy, lowy = np.max((np.max(m1), np.max(m2))), np.min((np.min(m1), np.min(m2)))
    axs["line1"].set_ylim(lowy-abs(0.1*lowy),upy+abs(0.1*upy))
    axs["line2"].set_ylim(lowy-abs(0.1*lowy),upy+abs(0.1*upy))


    sns.heatmap(corr,ax=axs["corr"], cbar = False, vmin=-1, vmax=1., cmap=cmap)
    axs["corr"].set_yticks([])
    axs["corr"].set_xticks([])

    sns.heatmap(tmp1, ax= axs["plot1"], cbar = False)
    axs["plot1"].set_yticks([])
    axs["plot1"].set_xticks([0,srange+1,2*srange+1], [-srange,0,srange])
    axs["plot1"].set_ylabel("n = {}".format(i+1))

    sns.heatmap(tmp2, ax = axs["plot2"], cbar = False)
    _ = axs["plot2"].set_yticks([])
    _ = axs["plot2"].set_xticks([0,srange+1,2*srange+1], [-srange,0,srange])


    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(axs["corr_legend"], cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal', ticklocation="top")
    _ = cb1.set_label('correlation', fontsize=15)

    if type(mutasome)!=type(None):
        mut = zscore(mutasome)
        mut_tmp = []

        for i, c in enumerate(pos):
            tmp = mut[c-srange:c+srange+1]
            if sense[i] == "+" and len(tmp) == 2*srange+1:
                mut_tmp.append(tmp)

            elif sense[i] == "-" and len(tmp) == 2*srange+1:
                mut_tmp.append(tmp[::-1])

            else:
                continue

        mut = np.mean(mut_tmp, axis =0)

        tax = axs["line2"].twinx()
        tax.plot(mut, color = "red", alpha = 0.3)
        tax.tick_params(axis='y', colors = 'red')
        tax.yaxis.tick_right()
        tax.set_ylabel("Mutasome score", color="red")
        
    if display:
        fig.show()

    if save:
        assert path!="", "please indicate a path"
        fig.savefig(path)
    
    return fig, axs


def vcorrcoef(X, Y):
    #Alex Westbrook
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = np.sum((X-Xm)*(Y-Ym), axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1)*np.sum((Y-Ym)**2, axis=1))
    r = r_num/r_den
    return r

def gauss(sigma, mu = 0, maxone = False):
    tmps = 1/(sigma*np.sqrt(2*np.pi))
    f = lambda x: 0.5*((x-mu)/sigma)**2
    g = np.array([tmps*np.exp(-f(x)) for x in range(-3*sigma, (3*sigma)+1)])
    if maxone:
        return g/np.max(g)
    else:
        return g