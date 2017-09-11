#!/usr/bin/env python3
from setuptools import setup

setup(
    name='bl3d',
    version='0.0.1',
    description='Cell segmentation in 3D GCaMP structural recordings',
    author='Fabian Sinz, Edgar Walker, Erick Cobos',
    author_email='ecobos@bcm.edu',
    license='MIT',
    url='https://github.com/cajal/bl3d',
    keywords= '2p 3d GCaMPs soma segmentation',
    packages=['bl3d'],
    install_requires=['numpy', 'torch'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English'
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
