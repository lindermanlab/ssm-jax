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
        "jax==0.4.8",
        "jaxlib==0.4.7+cuda11.cudnn86",
        "jupyter",
        "ipywidgets",
        "tensorflow-probability==0.17.0",
	    "flax==0.6.4",
 	    "optax",
    ],
    packages=setuptools.find_packages(),
)
