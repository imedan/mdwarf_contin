from typing import Tuple
import numpy as np
import astropy.units as u
from importlib.resources import open_binary
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import scipy.stats as ss
import warnings

# below is extra dependency
try:
    from extinction import ccm89, apply
except ModuleNotFoundError:
    warn_message = ('Extra dependecy extinction not installed.'
                    'Will not have full functionality.')
    warnings.warn(warn_message)


# get response data
resp = np.load(open_binary('mdwarf_contin.response_data', 'resp_PCA_results.npz').name)
mean_resp = resp['mean']
cov_resp = resp['cov']
components = resp['components']

try:
    with open(open_binary('mdwarf_contin.response_data', 'ivar_RF_model.pkl').name, 'rb') as f:
        rf_ivar = pickle.load(f)
        # turn off verbose
        rf_ivar.verbose = 0
        # only use 1 core
        rf_ivar.n_jobs = 1
except FileNotFoundError:
    rf_ivar = None
    warn_message = ('File ivar_RF_model.pkl not in mdwarf_contin.response_data. '
                    'Cannot simulate errors for SDSS-like spectra.')
    warnings.warn(warn_message)



def add_reddening(loglam: np.ndarray, flux: np.ndarray,
                  av: float) -> np.ndarray:
    """
    add reddening to a spectrum

    Parameters
    ----------
    loglam: np.array
        log of the wavelength of the spectrum

    flux: np.array
        Flux of the spectrum

    av: float
        The level of extinction

    Return
    ------
    flux_red: np.array
        reddened flux values of the spectrum
    """
    loglam = np.array(loglam, dtype=float)
    flux = np.array(flux, dtype=float)
    flux_red = apply(ccm89((10 ** loglam) * u.AA, av, 3.1), flux)
    return flux_red


def random_response(loglam: np.ndarray, flux: np.ndarray,
                    RNG: np.random._generator.Generator = np.random.default_rng(666)) -> Tuple[np.ndarray,
                                                                                               np.ndarray]:
    """
    Add random response to flux based on eigen vectors from
    difference spectra

    Parameters
    ----------
    loglam: np.array
        log of the wavelength of the spectrum

    flux: np.array
        Flux of the spectrum

    RNG: np.random._generator.Generator
        random state

    Return
    ------
    flux_resp: np.array
        flux values with random response added

    rand_weights: np.array
        weights for each PCA component applied to the spectrum
    """
    rand_weights = RNG.multivariate_normal(mean_resp, cov_resp)
    for i in range(len(components)):
        if i == 0:
            resp = components[i] * rand_weights[i]
        else:
            resp += components[i] * rand_weights[i]
    # normalize data like difference spectra were
    mask = (7495 <= 10 ** loglam) * (10 ** loglam <= 7505)
    med = np.nanmedian(flux[mask])
    flux_resp = (flux / med + resp) * med
    return flux_resp, rand_weights


def random_ivar(loglam: np.ndarray, flux: np.ndarray,
                RNG: np.random._generator.Generator = np.random.default_rng(666)) -> np.ndarray:
    """
    Add random response to flux based on eigen vectors from
    difference spectra

    Parameters
    ----------
    flux: np.array
        Flux of the spectrum

    snr: float
        desired SNR for output

    RNG: np.random._generator.Generator
        random state

    Return
    ------
    ivar: np.array
        The ivar for the spectrum given the snr
    """
    if rf_ivar is None:
        message = ('File ivar_RF_model.pkl not in mdwarf_contin.response_data. '
                   'Cannot simulate errors for SDSS-like spectra.')
        raise FileNotFoundError(message)
    mask = (7495 <= 10 ** loglam) * (10 ** loglam <= 7505)
    med = np.nanmedian(flux[mask])
    ivar = rf_ivar.predict(flux.reshape(-1, 1).T / med)
    ivar = ivar / med ** 2
    return ivar


def add_noise(flux: np.ndarray, snr: float,
              RNG: np.random._generator.Generator = np.random.default_rng(666)) -> np.ndarray:
    """
    Add nosie to signal

    Parameters
    ----------
    flux: np.array
        Flux of the spectrum

    snr: float
        desired SNR for output

    RNG: np.random._generator.Generator
        random state

    Return
    ------
    flux_noise: np.array
        flux values with noise added
    """
    flux_noise = RNG.normal(flux, flux / snr)
    return flux_noise


ckms = 2.998e5
sigma_to_fwhm = 2.355

def smooth_spec(wave: np.ndarray,
                spec: np.ndarray,
                outwave: np.ndarray,
                resolution: float,
                nsigma: float = 10,
                inres: float = 0) -> np.ndarray:
    """
    Smooth a spectrum to a constant resolution R.

    Below code is largely from the Payne, with license:

    The MIT License (MIT)

    Copyright (c) 2018 - Present: Phillip Cargile and contributors.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Parameters
    ----------
    wave: np.array
        Wavelength vector of the input spectrum (Angstroms).
    spec: np.array
        Flux vector of the input spectrum.
    outwave: np.array
        Desired output wavelength vector (Angstroms).
    resolution: float
        Desired resolving power, R, defined as lambda / delta_lambda
    nsigma: float
        Number of sigma away from the output wavelength to consider in the
        integral.  If less than zero, all wavelengths are used.  Setting this
        to some positive number decreses the scaling constant in the O(N_out *
        N_in) algorithm used here.
    inres: float
        The velocity resolution of the input spectrum (km/s), *not* FWHM.
    
    Returns
    --------
    flux: np.array
        Flux vector of the output spectrum at resolution R
    """
    Rsigma = resolution
    sigma = ckms / Rsigma
    fwhm = sigma * sigma_to_fwhm
    R = ckms / fwhm
    width = Rsigma

    sigma_eff_sq = sigma**2 - inres**2
    if np.any(sigma_eff_sq) < 0.0:
        raise ValueError("Desired velocity resolution smaller than the value"
                         "possible for this input spectrum.".format(inres))
    # sigma_eff is in units of sigma_lambda / lambda
    sigma_eff = np.sqrt(sigma_eff_sq) / ckms

    lnwave = np.log(wave)
    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (np.log(w) - lnwave) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * x ** 2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
    return flux


def manipulate_model_spectra(loglam_sdss: np.ndarray,
                             loglam_model: np.ndarray,
                             flux_model: np.ndarray,
                             size: int,
                             resolution: float = 2000.,
                             RNG: np.random._generator.Generator = np.random.default_rng(666),
                             calc_ivar: bool = True) -> Tuple[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray]:
    """
    Manipulate a model spectrum by by smoothing, downsampling
    adding reddening and instrument response

    Parameters
    ----------
    loglam_sdss: np.array
        loglam from sdss spectrum

    loglam_model: np.array
        loglam of the model spectrum

    flux_model: np.array
        flux of the model spectrum

    size: int
        number of random spectra to return

    resolution: float
        Desired resolving power, R, defined as lambda / delta_lambda

    RNG: np.random._generator.Generator
        random state

    calc_ivar: bool
        whether or not to calculate the IVAR.

    Returns
    -------
    flux_rand: np.array
        Flux of the random manipulated model spectra.
        Flux is evaluated at loglam_sdss. Size of array will be
        (size, len(loglam_sdss)).

    ivar_rand: np.array
        The IVAR for the random spectrum based on SNR applied
        to the spectrum.

    flux_smooth_down: np.array
        the model flux that has been smoothed and downsampled.
        Noisy spectrum should be compared to this.

    av_rand: np.array
        random extinction A_V values added to spectra

    snr: np.array
        SNR applied to the spectrum

    rand_weights: np.array
        weights for each PCA component applied to the spectrum
    """
    # check if RF model exists
    if rf_ivar is None:
        calc_ivar = False
        warn_message = ('File ivar_RF_model.pkl not in mdwarf_contin.response_data. '
                        'Cannot simulate errors for SDSS-like spectra.')
        warnings.warn(warn_message)

    # smooth and downsample the spectrum
    flux_smooth_down = smooth_spec(10 ** loglam_model,
                                   flux_model,
                                   10 ** loglam_sdss,
                                   resolution)

    flux_rand = np.zeros((size, len(loglam_sdss)))
    ivar_rand = np.zeros((size, len(loglam_sdss)))

    # add redenning to the spectra
    P = np.array([1.5402553, -0.0009273592438195921, 0.27507633])  # fit to 1 kpc M dwarfs
    av_rand = ss.lognorm.rvs(*P, size=size, random_state=RNG)
    for i in range(size):
        flux_rand[i, :] = add_reddening(loglam_sdss, flux_smooth_down, av_rand[i])

    # add noise to the spectrum
    # assume some uniform distriubtion for SNR
    snr = RNG.uniform(low=5, high=60, size=size)
    for i in range(size):
        flux_rand[i, :] = add_noise(flux_rand[i, :], snr[i], RNG=RNG)

    # add the instrument response
    rand_weights = np.zeros((size, len(components)))
    for i in range(size):
        flux_rand[i, :], rand_weights[i, :] = random_response(loglam_sdss, flux_rand[i, :], RNG=RNG)
        if calc_ivar:
            ivar_rand[i, :] = random_ivar(loglam_sdss, flux_rand[i, :], RNG=RNG)

    return flux_rand, ivar_rand, flux_smooth_down, av_rand, snr, rand_weights
