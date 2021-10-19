#!/usr/bin/env python
import os
from distutils.core import setup


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
        "jax",
        "jaxlib",
        "h5py",
        "jupyter",
        "ipywidgets",
        "tensorflow-probability",
    ],
    packages=["ssm"],
)
