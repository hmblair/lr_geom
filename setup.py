from setuptools import setup, find_packages

NAME = 'lr_geom'
VERSION = '0.1.0'
DESCRIPTION = 'Low-Rank Geometric Deep Learning'
AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/lr_geom'
LICENSE = 'MIT'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('README.md').read() if __import__('os').path.exists('README.md') else DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'torch',
    ],
    extras_require={
        'spherical': ['sphericart'],
        'graphs': ['dgl'],
        'all': ['sphericart', 'dgl'],
    },
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
