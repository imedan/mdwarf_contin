from typing import Union, Tuple, Callable
import numpy as np
from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from shapely.geometry.polygon import Polygon
from shapely import LineString, intersection, MultiLineString
from localreg.rbf import gaussian


def local_sigma_clip(x: np.ndarray, y: np.ndarray, x_range: tuple,
                     window: float = 200e-4,
                     sig: float = 5.) -> np.ndarray:
    """
    sigma clips in a moving window

    Parameters
    ----------
    x: np.array
        data where the binning will be done along

    y: np.array
        data to take median of
    
    x_range: np.array
        range of the data considered

    window: int
        size of the moving window (units of x)

    sig: float
        sigma to clip at

    Returns
    -------
    mask: np.array
        boolean mask of points that pass clipping
    """
    mask = np.zeros(len(x), dtype=bool) + True
    for i in range(len(x)):
        try:
            iend = np.where(x > x[i] + window)[0][0]
        except IndexError:
            iend = len(x)
        med = np.nanmean(x[i: iend])
        std = np.nanstd(x[i: iend])
        mask[i: iend][abs(x[i: iend] - med) > 5 * std] = False
    return mask


def median_filt(x: np.ndarray, y: np.ndarray, x_range: tuple,
                size: float, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    x_range: np.array
        range of the data considered

    size: float
        size of bins in (units of x)

    mask: np.array
        mask for the flux data

    Returns
    -------
    xm: np.array
        middle points of bins

    ym: np.array
        median in each of the x bins
    """
    bin_edges = np.arange(x_range[0], x_range[1] + size, size)
    xm = np.zeros(len(bin_edges) - 1) + np.nan
    ym = np.zeros(len(bin_edges) - 1) + np.nan
    for ind in range(len(xm)):
        xm[ind] = (bin_edges[ind] + bin_edges[ind + 1]) / 2
        ev = (x >= bin_edges[ind]) & (x < bin_edges[ind + 1])
        if np.sum(ev & mask) > 0:
            ym[ind] = np.nanmedian(y[ev & mask])
    # remove any nans
    ev = ~np.isnan(ym)
    return xm[ev], ym[ev]


def normalize_data(x: np.ndarray, y: np.ndarray,
                   mask: np.ndarray,
                   x_data_range: tuple = None, y_data_range: tuple = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the data prior to calculating alpha shape

    Parameters
    ---------
    x: np.array
        loglam data

    y: np.array
        flux data

    mask: np.array
        mask for the flux data

    x_data_range: tuple
        sets the min/max range for the data. If None, will use
        min/max of data provided

    y_data_range: tuple
        sets the min/max range for the data. If None, will use
        min/max of data provided

    Returns
    -------
    xn: np.array
        loglam data normalized

    yn: np.array
        flux data normalized
    """
    if x_data_range is None:
        xn = (x - np.nanmin(x[mask])) / (np.nanmax(x[mask]) - np.nanmin(x[mask]))
    else:
        xn = (x - x_data_range[0]) / (x_data_range[1] - x_data_range[0])
    if y_data_range is None:
        yn = (y - np.nanmin(y[mask])) / (np.nanmax(y[mask]) - np.nanmin(y[mask]))
    else:
        yn = (y - y_data_range[0]) / (y_data_range[1] - y_data_range[0])
    return xn, yn


def un_normalize_data(x: np.ndarray, xn: np.ndarray,
                      mask: np.ndarray, x_data_range: tuple = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Undo the normalize to the data

    Parameters
    ---------
    x: np.array
        the original data used for the normalization

    xn: np.array
        data normalized

    mask: np.array
        mask for the flux data

    x_data_range: tuple
        sets the min/max range for the data. If None, will use
        min/max of data provided

    Returns
    -------
    x0: np.array
        data unnormalized
    """
    if x_data_range is None:
        x0 = xn * (np.nanmax(x[mask]) - np.nanmin(x[mask])) + np.nanmin(x[mask])
    else:
        x0 = xn * (x_data_range[1] - x_data_range[0]) + x_data_range[0]
    return x0


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
                               (xa[i], 100)])
            xy = intersection(line, alpha_shape).xy
            xi = xy[0]
            yi = xy[1]
            if yi[np.argmax(yi)] == ya[i]:
                xmax.append(xa[i])
                ymax.append(ya[i])
        except (ValueError, NotImplementedError) as error:
            pass
    return np.array(xmax), np.array(ymax)


def localreg(x: np.ndarray, y: np.ndarray,
             x0: np.ndarray = None, degree: int = 2,
             kernel: Callable = gaussian, radius: float = 1.) -> np.ndarray:
    """
    rewrote localreg (https://github.com/sigvaldm/localreg/tree/master) function
    to improve speed

    Parameters
    ----------
    x: np.array
        x data for the fitting

    y: np.array
        y data for the fitting

    x0: np.array
        where the fit will be evalulated at for the output

    degree: int
        degree of the polynomial for the fit

    kernel: Callable
        kernel to apply to the weights

    radius: float
        value used for setting the weights at each point in x0.
        Acts as a smoothing factor

    Returns
    -------
    y0: np.array
        output of the regression at x0
    """
    if x0 is None:
        x0 = x

    if x.ndim == 1:
        x = x[:, np.newaxis]  # Reshape to 2D if it's 1D
    if x0.ndim == 1:
        x0 = x0[:, np.newaxis]  # Reshape to 2D if it's 1D

    n_samples, n_indeps = x.shape
    n_samples_out, _ = x0.shape

    y0 = np.zeros(n_samples_out)

    powers = np.arange(degree + 1)
    B = np.stack(np.meshgrid(*([powers] * n_indeps), indexing='ij'), axis=-1)
    
    X = np.prod(np.power(x[:, :, np.newaxis], B.T), axis=1)
    X0 = np.prod(np.power(x0[:, :, np.newaxis], B.T), axis=1)

    weights = kernel(np.linalg.norm(x[:, np.newaxis] - x0, axis=-1) / radius)
    s_weights = np.sqrt(weights)
    lhs0 = X[:, :, np.newaxis] * s_weights[:, np.newaxis, :]
    rhs = y[:, np.newaxis] * s_weights

    # need to do this to reshape things
    lhs = np.zeros((n_samples_out, n_samples, degree + 1))
    for i, xi in enumerate(x0):
        lhs[i, :, :] = lhs0[:, :, i]

    # Compute pseudo-inverse directly instead of using lstsq
    lhs_inv = np.linalg.pinv(lhs)
    for i, xi in enumerate(x0):
        beta = lhs_inv[i, :, :] @ rhs[:, i]
        y0[i] = X0[i, :] @ beta
    return y0


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

    size: float
        size of the bins (in log(Angstroms)) for the median filtering

    alpha: float
        alpha size for alpha hulling

    degree: int
        degree of polynomial for local regression

    kernel: Callable
        kernel to use for local polynomial regression

    radius: float
        smoothing parameter for local polynomial regression

    sigma_clip: bool
        Whether or not to sigma clip the flux data with a moving
        before processing the continuum

    loglam_range: tuple
        sets the min/max range for scaling the spectrum. If None,
        will use min/max of loglam provided

    flux_range: tuple
        sets the min/max range for scaling the spectrum. If None,
        will use min/max of flux provided

    aspect_ratio: float
        this will set the flux_range for a given aspect ratio.
        This is done on the median filter of the spectrum. Aspect
        ratio is defined as the normalized flux range / normalized loglam range

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
    def __init__(self, loglam: np.ndarray, flux: np.ndarray, size: int = 13e-4,
                 alpha: float = 12.477607, degree: int = 3, kernel: Callable = gaussian,
                 radius: float = 0.160439, sigma_clip: bool = True,
                 loglam_range: tuple = (3.6001, 4.017), flux_range: tuple = None,
                 aspect_ratio: float = 1.533087):
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
        self.loglam_range = loglam_range
        self.flux_range = flux_range
        self.aspect_ratio = aspect_ratio

        # set ranges based on max if None
        if self.loglam_range is None:
            self.loglam_range = (np.nanmin(self.loglam), np.nanmax(self.loglam))

        # get mask from sigma clipping
        if sigma_clip:
            self.mask = local_sigma_clip(self.loglam, self.flux, self.loglam_range)
        else:
            self.mask = np.zeros(len(self.flux), dtype=bool) + True

        # median filter the spectrum
        self.loglam_med, self.flux_med = median_filt(self.loglam, self.flux,
                                                     self.loglam_range,
                                                     size=self.size, mask=self.mask)

        # get flux_range if aspect ratio set
        if aspect_ratio is not None:
            norm_fact = ((np.nanmax(self.flux_med) - np.nanmin(self.flux_med)) / self.aspect_ratio + np.nanmin(self.flux_med)) / np.nanmax(self.flux_med)
            self.flux_range = (np.nanmin(self.flux_med), np.nanmax(self.flux_med) * norm_fact)

        # set ranges based on max if None
        if self.flux_range is None:
            self.flux_range = (np.nanmin(self.flux[self.mask]), np.nanmax(self.flux[self.mask]))   

        # normalize the data
        self.loglam_norm, self.flux_norm = normalize_data(self.loglam, self.flux,
                                                          self.mask, x_data_range=self.loglam_range,
                                                          y_data_range=self.flux_range)

        # normalize data the median filter data
        self.loglam_med, self.flux_med = normalize_data(self.loglam_med, self.flux_med,
                                                        np.zeros(len(self.loglam_med), dtype=bool) + True,
                                                        x_data_range=self.loglam_range,
                                                        y_data_range=self.flux_range)

    def find_continuum(self):
        """
        find the continuum by calculating alpha hull and doing
        local polynomial regression
        """
        self.alpha_shape = calculate_alpha_shape(self.loglam_med, self.flux_med,
                                                 alpha=self.alpha)
        self.loglam_max, self.flux_max = max_intersect(self.alpha_shape)
        self.continuum_norm = localreg(self.loglam_max, self.flux_max,
                                       x0=self.loglam_norm, degree=self.degree,
                                       kernel=self.kernel, radius=self.radius)
        self.continuum = un_normalize_data(self.flux,
                                           self.continuum_norm,
                                           self.mask,
                                           x_data_range=self.flux_range)
