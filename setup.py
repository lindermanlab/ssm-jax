#!/usr/bin/env python
from distutils.core import setup
import setuptools
import os


setup(
    name="ssm",
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
        "jax==0.3.20",
        "jaxlib==0.3.20",
        "jupyter",
        "ipywidgets",
        "tensorflow-probability==0.16.0",
	"flax==0.5.2",
 	"optax",
        "orbax==0.1.0",
        "orbax-checkpoint==0.1.1"
    ],
    packages=setuptools.find_packages(),
)
