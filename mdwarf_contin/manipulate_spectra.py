from extinction import ccm89, apply
import numpy as np
import astropy.units as u
from importlib.resources import open_binary
import pickle


# get PCA data
with open(open_binary('mdwarf_contin.response_data', 'pca_diff_spectra.pkl').name, 'rb') as f:
    res = pickle.load(f)
    components = res[0]
    X_projected = res[1]


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


def random_response(loglam: np.ndarray, flux: np.ndarray):
    """
    Add random response to flux based on eigen vectors from
    difference spectra

    Parameters
    ----------
    loglam: np.array
        log of the wavelength of the spectrum

    flux: np.array
        Flux of the spectrum

    Return
    ------
    flux_resp: np.array
        flux values with random response added
    """
    for i in range(len(components)):
        if i == 0:
            resp = components[i] * np.random.choice(X_projected[:, i], 1)[0]
        else:
            resp += components[i] * np.random.choice(X_projected[:, i], 1)[0]
    # normalize data like difference spectra were
    mask = (7495 <= 10 ** loglam) * (10 ** loglam <= 7505)
    med = np.nanmedian(flux[mask])
    flux_resp = (flux / med + resp) * med
    return flux_resp
