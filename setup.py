from setuptools import setup, find_packages
import os


setup(
    name='interpNLTE',
    version='0.0.1',
    description='NLTE interpolation package',
    packages=find_packages(),  # Automatically discover packages and subpackages
    install_requires=[
        # List your package dependencies here, e.g., 'requests>=2.0'
    ],
)