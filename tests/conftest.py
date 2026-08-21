from pathlib import Path
import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def real_spectrum():
    """loglam/flux arrays extracted from a real SDSS M dwarf spectrum
    (spec-104193-60087-27021598685558821.fits, LOGLAM > 3.6), stored
    as a small .npz fixture instead of the original FITS file.
    """
    with np.load(DATA_DIR / "example_spectrum.npz") as data:
        return data["loglam"], data["flux"]
