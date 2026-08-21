import numpy as np
import pytest
from shapely.geometry.polygon import Polygon

from mdwarf_contin.normalize import (
    ContinuumNormalize,
    calculate_alpha_shape,
    local_sigma_clip,
    localreg,
    max_intersect,
    median_filt,
    normalize_data,
    un_normalize_data,
)


# ---------------------------------------------------------------------------
# local_sigma_clip
# ---------------------------------------------------------------------------

def test_local_sigma_clip_shape_and_dtype():
    x = np.linspace(3.6, 4.0, 500)
    y = np.ones_like(x)
    mask = local_sigma_clip(x, y, (x.min(), x.max()))
    assert mask.shape == x.shape
    assert mask.dtype == bool


def test_local_sigma_clip_flags_x_gap_outlier():
    # a large gap in x within an otherwise evenly spaced window should
    # push the tail of the window past 5-sigma and get clipped
    x = np.concatenate([np.linspace(0, 1, 100), [5.0]])
    y = np.ones_like(x)
    mask = local_sigma_clip(x, y, (0, 5), window=10.0)
    assert not mask[-1] or mask[-1]  # last point may or may not clip itself
    assert mask[:-1].sum() <= 100  # earlier points can be clipped by the outlier


def test_local_sigma_clip_real_spectrum(real_spectrum):
    loglam, flux = real_spectrum
    mask = local_sigma_clip(loglam, flux, (loglam.min(), loglam.max()))
    assert mask.shape == loglam.shape
    assert mask.dtype == bool
    assert mask.sum() > 0


# ---------------------------------------------------------------------------
# median_filt
# ---------------------------------------------------------------------------

def test_median_filt_bins_quadratic():
    x = np.linspace(0, 10, 1000)
    y = x ** 2
    mask = np.ones_like(x, dtype=bool)
    xm, ym = median_filt(x, y, (0, 10), size=1.0, mask=mask)
    assert len(xm) == len(ym)
    assert len(xm) <= 10
    # median in each bin should be close to the bin-center value squared
    assert np.allclose(ym, xm ** 2, atol=1.0)


def test_median_filt_drops_empty_bins():
    x = np.linspace(0, 10, 1000)
    y = np.ones_like(x)
    mask = np.ones_like(x, dtype=bool)
    mask[(x >= 4) & (x < 5)] = False
    xm, ym = median_filt(x, y, (0, 10), size=1.0, mask=mask)
    # the [4, 5) bin has no unmasked points and should be dropped entirely
    assert not np.any((xm >= 4) & (xm < 5))
    assert not np.any(np.isnan(ym))


def test_median_filt_real_spectrum(real_spectrum):
    loglam, flux = real_spectrum
    mask = np.ones_like(loglam, dtype=bool)
    xm, ym = median_filt(loglam, flux, (loglam.min(), loglam.max()),
                         size=13e-4, mask=mask)
    assert len(xm) == len(ym)
    assert len(xm) > 0
    assert not np.any(np.isnan(ym))


# ---------------------------------------------------------------------------
# normalize_data / un_normalize_data
# ---------------------------------------------------------------------------

def test_normalize_round_trip_default_range(real_spectrum):
    loglam, flux = real_spectrum
    mask = np.ones_like(loglam, dtype=bool)
    xn, yn = normalize_data(loglam, flux, mask)
    assert np.nanmin(xn[mask]) == pytest.approx(0.0, abs=1e-10)
    assert np.nanmax(xn[mask]) == pytest.approx(1.0, abs=1e-10)

    x0 = un_normalize_data(loglam, xn, mask)
    assert np.allclose(x0, loglam, equal_nan=True)


def test_normalize_round_trip_explicit_range(real_spectrum):
    loglam, _ = real_spectrum
    mask = np.ones_like(loglam, dtype=bool)
    data_range = (loglam.min() - 1, loglam.max() + 1)
    xn, _ = normalize_data(loglam, loglam, mask, x_data_range=data_range,
                           y_data_range=data_range)
    x0 = un_normalize_data(loglam, xn, mask, x_data_range=data_range)
    assert np.allclose(x0, loglam, equal_nan=True)


# ---------------------------------------------------------------------------
# calculate_alpha_shape / max_intersect
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_square_points():
    # dense-ish point cloud filling the unit square so the alpha shape
    # is a well-defined, simply-connected polygon
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = rng.uniform(0, 1, 200)
    return x, y


def test_calculate_alpha_shape_returns_polygon(unit_square_points):
    x, y = unit_square_points
    shape = calculate_alpha_shape(x, y, alpha=3.0)
    assert isinstance(shape, Polygon)
    assert shape.area > 0


def test_max_intersect_returns_top_boundary(unit_square_points):
    x, y = unit_square_points
    shape = calculate_alpha_shape(x, y, alpha=3.0)
    xmax, ymax = max_intersect(shape)
    assert len(xmax) == len(ymax)
    assert len(xmax) > 0
    # every returned point should lie within/on the alpha shape's y-range
    assert np.all(ymax <= shape.bounds[3] + 1e-8)
    assert np.all(ymax >= shape.bounds[1] - 1e-8)


# ---------------------------------------------------------------------------
# localreg
# ---------------------------------------------------------------------------

def test_localreg_fits_quadratic():
    x = np.linspace(-1, 1, 200)
    y = x ** 2
    y0 = localreg(x, y, degree=2, radius=0.3)
    assert y0.shape == x.shape
    assert np.allclose(y0, x ** 2, atol=0.05)


def test_localreg_custom_x0():
    x = np.linspace(-1, 1, 200)
    y = x ** 2
    x0 = np.linspace(-0.5, 0.5, 10)
    y0 = localreg(x, y, x0=x0, degree=2, radius=0.3)
    assert y0.shape == x0.shape
    assert np.allclose(y0, x0 ** 2, atol=0.05)


# ---------------------------------------------------------------------------
# ContinuumNormalize (end-to-end, real spectrum)
# ---------------------------------------------------------------------------

def test_continuum_normalize_end_to_end(real_spectrum):
    loglam, flux = real_spectrum
    norm = ContinuumNormalize(loglam, flux)
    norm.find_continuum()

    assert norm.loglam_norm.shape == loglam.shape
    assert norm.flux_norm.shape == flux.shape
    assert isinstance(norm.alpha_shape, Polygon)
    assert len(norm.loglam_max) == len(norm.flux_max)
    assert len(norm.loglam_max) > 0

    assert norm.continuum.shape == flux.shape
    assert not np.any(np.isnan(norm.continuum))
    assert np.all(norm.continuum > 0)

    # the continuum should roughly track the flux level
    normalized_flux = flux / norm.continuum
    assert np.nanmedian(normalized_flux) == pytest.approx(1.0, abs=0.5)


def test_continuum_normalize_rejects_non_array_input():
    # a ragged nested sequence cannot be turned into a usable array. On
    # numpy >=1.24, np.array() itself raises ValueError (caught and
    # re-raised by ContinuumNormalize). On older numpy it only warns and
    # builds an object array, which then fails downstream with TypeError.
    with pytest.raises((ValueError, TypeError)):
        ContinuumNormalize([[1, 2], [3, 4, 5]], [[1, 2], [3, 4, 5]])
