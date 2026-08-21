# Continuum Normalization of M Dwarfs

[![tests](https://github.com/imedan/mdwarf_contin/actions/workflows/tests.yml/badge.svg)](https://github.com/imedan/mdwarf_contin/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16921357.svg)](https://doi.org/10.5281/zenodo.16921357)
[![Paper](https://img.shields.io/badge/DOI-10.3847%2F1538--3881%2Fae0a12-blue)](https://doi.org/10.3847/1538-3881/ae0a12)

Code to continuum normalize M dwarfs using alpha hulls and local polynomial regression. This package is based on the methods outlined in [Medan & Way et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025AJ....170..302M/abstract).

## Installation

Use the following commands to install the code locally.

```
git clone https://github.com/imedan/mdwarf_contin
cd mdwarf_contin
conda create -n "mdwarf_contin_code" python=3.10 ipython
conda activate mdwarf_contin_code
pip install poetry
poetry install --without extras
```

If you want to install the extra dependencies needed to use features that manipulate model spectra 
into SDSS-like spectra, then the extra dependencies must also be installed. This can be done by running:

```
poetry install
```

## Usage

The notebook in [`tests/example_usage.ipynb`](tests/example_usage.ipynb) provides an overview on how to use the code for doing normalization.
