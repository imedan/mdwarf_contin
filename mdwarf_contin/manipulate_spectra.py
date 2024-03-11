from extinction import ccm89, apply
import numpy as np
import astropy.units as u


def add_reddening(loglam: np.ndarray, flux: np.ndarray, av: float):
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
