from setuptools import setup
from graphtime import __version__
setup(
    name='graphtime',
    version = __version__,
    author='Simon Olsson',
    author_email='simon.olsson@fu-berlin.de',
    packages=['graphtime', 'graphtime.test'],
    scripts=[],
    url='http://127.0.0.1',
    license='LICENSE.txt',
    description='A module for learning encoding of transition probabilities with undirected graphical models',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.3",
        "scikit-learn >= 0.19.0",
        "scipy >= 1.1.0",
        "msmtools >= 1.2.1",
    ],
)