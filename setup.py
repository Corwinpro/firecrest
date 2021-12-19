"""
Setup configuration for installation via pip
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()


# The find_packages function does a lot of the heavy lifting for us w.r.t.
# discovering any Python packages we ship.
setup(
    name="firecrest",
    version="0.1.0.dev0",
    packages=find_packages(),
    # PyPI packages required for the *installation* and usual running of the
    # tools.
    install_requires=required,
    entry_points={"console_scripts": ["firecrest = firecrest.cli.firecrest:run"]},
    # Metadata for PyPI (https://pypi.python.org).
    description="Finite Element shape optimization tool based on FEniCS",
)
