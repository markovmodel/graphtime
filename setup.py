from distutils.core import setup

setup(
    name='graphtime',
    version='0.0a',
    author='Simon Olsson',
    author_email='simon.olsson@fu-berlin.de',
    packages=['graphtime', 'graphtime.test'],
    scripts=[],
    url='http://127.0.0.1',
    license='LICENSE.txt',
    description='A module for learning encoding of transition probabilities with undirected graphical models',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.3",
        "scikit-learn >= 0.19.0",
    ],
)

