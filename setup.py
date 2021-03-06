import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "RobustRAQN",
    version = "0.0.1",
    author = "Felix Halpaap",
    author_email = "felix.halpaap@uib.no",
    description = ("Demonstration of robust request and seismic data quality "
                   " metrics negotiation."),
    license = "GPL3",
    keywords = "example documentation tutorial",
    # url = "http://packages.python.org/an_example_pypi_project",
    packages=['robustraqn'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Library or General Public '
        'License (GPL)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
