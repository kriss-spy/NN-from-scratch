# setup.py
from setuptools import setup, find_packages

setup(
    name="my_neural_net",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.24.0", "tensorflow>=2.15.0"],
)
