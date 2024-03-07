from typing import Union, Tuple, Callable
import numpy as np
from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from shapely.geometry.polygon import Polygon
from shapely import LineString, intersection, MultiLineString
from localreg import *


def median_filt(x: np.ndarray, y: np.ndarray,
                size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a median filter to a set of data.
    Returns the medians is bins equal to size along
    the x direction.

    Parameters
    ----------
    x: np.array
        data where the binning will be done along

    y: np.array
        data to take median of

    size: int
        size of bins

    Returns
    -------
    xm: np.array
        middle points of bins

    ym: np.array
        median in each of the x bins
    """
    xm = np.zeros(len(x) // size - 1)
    ym = np.zeros(len(x) // size - 1)
    for ind in range(len(xm)):
        xm[ind] = (x[ind * size] + x[(ind + 1) * size]) / 2
        ym[ind] = np.nanmedian(y[ind * size: (ind + 1) * size])
    return xm, ym


def normalize_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the data prior to calculating alpha shape

    Parameters
    ---------
    x: np.array
        loglam data

    y: np.array
        flux data

    Returns
    -------
    xn: np.array
        loglam data normalized

    yn: np.array
        flux data normalized
    """
    xn = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    yn = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
    return xn, yn


def calculate_alpha_shape(x: np.ndarray, y: np.ndarray,
                          alpha: float = 1 / 0.05) -> Polygon:
    """
    Calculate the alpha shape for a spectrum

    Parameters
    ---------
    x: np.array
        loglam data that has been normalized

    y: np.array
        flux data that has been normalized

    alpha: float
        alpha value to use for alpha hull. If None
        then will pick optimal value

    Returns
    -------
    alpha_shape: Polygon
        alpha shape of the spectrum
    """
    shaper = Alpha_Shaper(np.column_stack((x, y)), normalize=False)
    if alpha is None:
        alpha_opt, alpha_shape = shaper.optimize()
    else:
        alpha_shape = shaper.get_shape(alpha=alpha)
    return alpha_shape


def max_intersect(alpha_shape: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the points at the maximum of an alpha shape

    Parameters
    ---------
    alpha_shape: Polygon
        alpha shape of the spectrum

    Returns
    -------
    xmax: np.array
        x coordinates of the alpha shape that intersect the maximum

    ymax: np.array
        y coordinates of the alpha shape that intersect the maximum
    """
    
    if type(alpha_shape.boundary)==LineString:
        xa = alpha_shape.boundary.xy[0]
        ya = alpha_shape.boundary.xy[1]
    elif type(alpha_shape.boundary)==MultiLineString:
        xa = np.array([])
        ya = np.array([])
        for foo in alpha_shape.boundary.geoms:
            xa = np.append(xa, np.array(foo.xy[0]))
            ya = np.append(ya, np.array(foo.xy[1]))
    else:
        raise ValueError("Help! Something has gone terribly wrong with the alpha_shape!")
    xmax = []
    ymax = []
    for i in range(len(xa)):
        try:
            line = LineString([(xa[i], 0),
                               (xa[i], 1)])
            xy = intersection(line, alpha_shape).xy
            xi = xy[0]
            yi = xy[1]
            if yi[np.argmax(yi)] == ya[i]:
                xmax.append(xa[i])
                ymax.append(ya[i])
        except NotImplementedError:
            pass
    return np.array(xmax), np.array(ymax)

class ContinuumNormalize(object):
    """
    Continuum normalize a spectrum using alpha hulls
    and local polynomial regression

    Parameters
    ----------
    loglam: np.array
        log of the wavelength of the spectrum

    flux: np.array
        Flux of the spectrum

    size: int
        size of the bins (in indicies) for the median filtering

    alpha: float
        alpha size for alpha hulling

    degree: int
        degree of polynomial for local regression

    kernel: Callable
        kernel to use for local polynomial regression

    radius: float
        smoothing parameter for local polynomial regression

    Attributes
    ----------
    loglam_norm: np.array
        log of the wavelength of the spectrum - normalized

    flux_norm: np.array
        Flux of the spectrum - normalized

    loglam_med: np.array
        log of the wavelength of the spectrum - normalized and median filtered

    flux_med: np.array
        Flux of the spectrum - normalized and median filtered

    alpha_shape: Polygon
        alpha shape of the spectrum

    loglam_max: np.array
        log of the wavelength of the spectrum - maximum alpha shape points

    flux_max: np.array
        Flux of the spectrum - maximum alpha shape points

    continuum: np.array
        the contimuum determined from fitting alpha shape max values
        with local polynomial regression
    """
    def __init__(self, loglam: np.ndarray, flux: np.ndarray, size: int = 7,
                 alpha: float = 1 / 0.05, degree: int = 2, kernel: Callable = rbf.tricube,
                 radius: float = 0.2):
        try:
            self.loglam = np.array(loglam)
            self.flux = np.array(flux)
        except:
            raise ValueError("loglam and flux must be 1d arrays")
         
        self.size = size
        self.alpha = alpha
        self.degree = degree
        self.kernel = kernel
        self.radius = radius

        # normalize the data
        self.loglam_norm, self.flux_norm = normalize_data(self.loglam, self.flux)

        # median filter the normalized data
        self.loglam_med, self.flux_med = median_filt(self.loglam_norm, self.flux_norm,
                                                     size=self.size)

    def find_continuum(self):
        """
        find the continuum by calculating alpha hull and doing
        local polynomial regression
        """
        self.alpha_shape = calculate_alpha_shape(self.loglam_med, self.flux_med,
                                                 alpha=self.alpha)
        self.loglam_max, self.flux_max = max_intersect(self.alpha_shape)
        self.continuum = localreg(self.loglam_max, self.flux_max,
                                  x0=self.loglam_norm, degree=self.degree,
                                  kernel=self.kernel, radius=self.radius)
