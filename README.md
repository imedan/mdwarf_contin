# Continuum Normalization of M Dwarfs

Code to continuum normalize M dwarfs using alpha hulls and local polynomial regression.

## Installation

Use the following commands to install the code locally.

```
git clone https://github.com/imedan/mdwarf_contin
cd mdwarf_contin
conda create -n "mdwarf_contin_code" python=3.10 ipython
conda activate mdwarf_contin_code
pip install poetry
poetry install
```

## Usage

The notebook in [`tests/example_usage.ipynb`](https://github.com/imedan/mdwarf_contin/blob/main/tests/example_usage.ipynb) provides an overview on the to use the code for doing normalization.
