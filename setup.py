#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import setuptools
from setuptools import setup, find_packages


packagename = "pxrd_simulator"  # this has to be renamed

# consider the path of `setup.py` as root directory:
PROJECTROOT = os.path.dirname(sys.argv[0]) or "."
release_path = os.path.join(PROJECTROOT, "src", packagename, "release.py")
with open(release_path, encoding="utf8") as release_file:
    __version__ = release_file.read().split('__version__ = "', 1)[1].split('"', 1)[0]


with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

setup(
    name=packagename,
    version=__version__,
    author="Artem Dmitrienko",
    author_email="dmitrienka@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    # url="https://codeberg.org/username/reponame",
    license="GPLv3",
    description="some description",
    long_description="""
    ...
    """,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
    entry_points={"console_scripts": [f"{packagename}={packagename}.script:main"]},
    include_package_data=True,
    package_data={'': ['data/*.npy']},
)
