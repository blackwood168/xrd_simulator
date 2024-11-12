# PXRD Simulator

A Python package for generating synthetic powder X-ray diffraction (PXRD) data and crystal structures for machine learning applications.

![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Overview

This repository provides tools for:
- Generating random crystal structures with specified parameters
- Calculating structure factors and reflection intensities 
- Simulating powder XRD patterns with realistic peak profiles and backgrounds
- Creating large datasets for machine learning applications

## Features

### Crystal Structure Generation
- Supports multiple space groups (P-1, P21/c, C2/c, etc.)
- Configurable atomic composition (C, N, O, Cl, etc.)
- Adjustable unit cell parameters and atom counts
- Based on the CCTBX crystallographic toolkit

### PXRD Pattern Simulation
- Thomson-Cox-Hastings pseudo-Voigt peak profiles
- Axial divergence corrections
- Chebyshev polynomial backgrounds
- Supports Cu Kα1 and Cu Kα1,2 radiation
- Poisson noise simulation


## Installation
```
git clone https://github.com/username/pxrd_simulator.git
pip install -r requirements.txt
```

### Dependencies
- numpy
- scipy
- cctbx
- pandas
- multiprocessing

## Scripts

### run.py
Generates powder XRD patterns with:
- Random crystal structures
- Realistic peak profiles
- Chebyshev polynomial backgrounds
- Multiple phases support (in progress?)
- Parallel processing capabilities

### structure_gen.py
Generates crystal structures and calculates their diffraction properties:
- Random atom positions
- Specified space groups
- Structure factors calculation
- Reflection intensities
- High/low resolution data

## Usage
### Script Execution
#### Generate Crystal Structures and Their Diffraction Data:
```
# Navigate to the package directory
cd src/pxrd_simulator

# Run structure generation script
python structure_gen.py
```
This will generate .npz files containing:
- Structure parameters (unit cell, space group, etc.)
- Miller indices for low/high resolution
- Structure factor intensities
- Output is saved to clin_test_100k/clin{d_low}_{d_high}_{i}.npz
### Generate PXRD Patterns:
```
# Run pattern generation script
python run.py
```
This generates:
- test_x_{i}.csv: PXRD patterns
- test_y_{i}.csv: Generation parameters

### Parallel Processing Configuration
```
# In structure_gen.py or run.py
CHANKS = 100        # Number of chunks to process
CHANK_SIZE = 10000  # Structures per chunk
CORES = 8           # Number of CPU cores to use
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.