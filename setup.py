#!/usr/bin/env python3
"""CorrAI"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="corrai",
    version="0.1.0",
    description="Data correction and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BuildingEnergySimulationTools/corrai",
    author="Nobatek/INEF4",
    author_email="bdurandestebe@nobatek.inef4.com",
    license="License :: OSI Approved :: BSD License",
    # keywords=[
    # ],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.3",
        "pandas>=1.3.4",
        "scipy>=1.7.2",
        "matplotlib>=3.5.1",
        "plotly>=5.3.1",
        "scikit-learn>=1.2.2",
        "pymoo>=0.6.0.1",
        "salib>=1.4.7",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
