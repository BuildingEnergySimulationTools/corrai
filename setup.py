#!/usr/bin/env python3
"""CorrAI"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="corrai",
    version="0.1",
    description="Data correction and Machine Learning",
    long_description=long_description,
    # url="",
    author="Nobatek/INEF4",
    author_email="bdurandestebe@nobatek.inef4.com",
    # license="",
    # keywords=[
    # ],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
