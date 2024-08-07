#!/usr/bin/env python3
"""CorrAI"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="corrai",
    version="0.3.1",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22.4, <2.0",
        "pandas>=2.1.0, <3.0",
        "scipy>=1.9.1, <2.0",
        "scikit-learn>=1.2.2, <2.0",
        "pymoo>=0.6.0.1",
        "salib>=1.4.7",
        "fmpy>=0.3.6",
        "matplotlib>=3.5.1",
        "plotly>=5.3.1",
        "fastprogress>=1.0.3",
        "keras>=2.14.0",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
