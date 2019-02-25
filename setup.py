#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='voxelize',
    version='0.1.0-dev.0',
    description='Speedy, simple, and efficient voxelization of meshes.',
    author='Christos Papadopoulos',
    author_email='cpapa97@gmail.com',
    packages=['voxelize'], #find_packages(exclude=[]), # I can't call the folder voxelize honestly. I use that word more than once.
    install_requires=['numpy', 'ipyvolume']
)
