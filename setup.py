from codecs import open  # To use a consistent encoding
from os import path

from setuptools import find_packages, setup  # Always prefer setuptools over distutils

here = path.abspath(path.dirname(__file__))

setup(
    name="itsml",
    version="0.1.0",
    description="Tools for doing machine learning with some of my custom transformers.",
    author="Alexandre Farias <0800alefarias@gmail.com>",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    test_suite="tests",
    install_requires=[],
    entry_points={},
)