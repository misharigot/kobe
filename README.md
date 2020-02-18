# kobe
Code base for the AI Bachelor course Machine Learning at the Vrije Universiteit Amsterdam. Predicting scoring shots using 20 years of data on Kobe Bryant's swishes and misses. See the [competition and data set on Kaggle](https://www.kaggle.com/c/kobe-bryant-shot-selection/overview).


## Installation with Poetry

This repository optionally uses [poetry](https://python-poetry.org/docs/) as a package/dependency manager.

If you wish to install the packages manually, go to the section [manual package installation](#manual-package-installation).

To set up your environment with Poetry:

```bash
poetry install
```

To create a kernel so you can use the env in your jupyter notebooks:

```bash
poetry run ipython kernel install --user --name=kobe
```

## Manual package installation

You can install the packages manually if you do not want to use Poetry. For a list of packages, see [pyproject.toml](pyproject.toml) under `[tool.poetry.dependencies]`.
