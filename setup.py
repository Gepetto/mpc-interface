#!/usr/bin/env python

from pathlib import Path
from setuptools import setup

PACKAGE_NAME = "qp_formulations"

setup(
    name=PACKAGE_NAME,
    version="1.0",
    description="Gecko.",
    author="Nahuel Villa",
    author_email="nvilla@laas.fr",
    package_dir={
        "formulation": str(Path(".") / "formulation"),
        "gecko": str(Path(".") / "gecko")         
    },
    packages=["formulation", "gecko"],
)