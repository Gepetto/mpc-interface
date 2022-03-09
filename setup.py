#!/usr/bin/env python

from pathlib import Path
from setuptools import setup

PACKAGE_NAME = "qp_formulations"
CURRENT_DIR = Path(__file__).resolve().parent

setup(
    name=PACKAGE_NAME,
    version="1.0",
    description="Gecko.",
    author="Nahuel Villa",
    author_email="nvilla@laas.fr",
    packages=["python/formulation", "python/gecko"],
)