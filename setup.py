#!/usr/bin/env python
from distutils.core import setup
import setuptools
import os


setup(
    name="ssm_jax",
    version="0.1",
    description="Bayesian learning and inference for a variety of state space models",
    author="Scott Linderman",
    author_email="scott.linderman@stanford.edu",
    url="https://github.com/lindermanlab/ssm",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "seaborn",
        "jax",
        "jaxlib",
        "jupyter",
        "ipywidgets",
        "tensorflow-probability",
	"flax",
 	"optax",
    ],
    packages=setuptools.find_packages(),
)
