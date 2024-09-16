[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# General Information


This repository holds the basic directory layout for new python projects. Additional to a reasonable minimum of features (setup.py, requirements.txt, directory structure, ...) it contains the following extras:

- basic unittest
- `script.py` and associated entrypoint in `setup.py` (allows to call some functionality of the package directly from command line (try `package_name cmd1`))


## Usage

- Rename directory `src/package_name`
- Edit `setupy.py`: replace dummy data with real data.
- Add your source. a) Either to [`core.py`](src/package_name/core.py) or b) to your own separate file(s).
    - a) simplifies importing your module
    - b) is more flexible but you have to take care of importability yourself.


For local development it is recommended to install this (better: your) package in [editable mode](https://pip.pypa.io/en/latest/cli/pip_wheel/?highlight=editable#cmdoption-e): `pip install -e .` (run from where `setup.py` lives).


## Publishing on pypi

To publish your package on [Python Package Index (pypi)](pypi.org/) the following commands should work:

- `python setup.py sdist`
- `twine upload  --repository-url https://upload.pypi.org/legacy/ -u <username> dist/*`
    - You need to register a username first.
    - There is probably a more modern way for uploading to pypi.
