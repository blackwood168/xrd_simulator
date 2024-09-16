#!/usr/bin/env python3

import core
import utils

from utils import GenBuilder, distr_gen, sample_gen

import numpy as np
import scipy.stats as sts
import itertools as it

# background

BKG_MAX_ORDER = 13
BKG_MIN_ORDER = 2

bkg_generator = GenBuilder(
    classname = core.ChebyshevBkg,
    coefs = distr_gen(sts.norm(), size = BKG_MAX_ORDER),
    order = distr_gen(sts.randint(BKG_MIN_ORDER, BKG_MAX_ORDER)))

# structure

SPACE_GROUPS = ["P-1", "P21/c", "C2/c", "Pbca", "I41"]
ELEMENTS = ["C", "N", "O", "Cl"]
N_ATOMS_LIMS = (3, 30)
ATOM_VOLUME_START_WIDTH = (14, 8)

n_atoms = sample_gen(range(*N_ATOMS_LIMS))

str_generator = GenBuilder(
    classname= core.CctbxStr.generate_packing,
    sg = sample_gen(SPACE_GROUPS),
    atoms = sample_gen(ELEMENTS, size = n_atoms),
    atom_volume = distr_gen( sts.uniform(*ATOM_VOLUME_START_WIDTH)),
    seed = utils.distr_gen(sts.randint(1, 2**32-1)))

# profiles

GAUSS_STEPS = 0.01

symmetric_profile_generator = GenBuilder(
    classname = core.PV_TCHZ,
    U = distr_gen(sts.uniform(1, 5)),
    V = distr_gen(sts.uniform(-1, 1)),
    W = distr_gen(sts.uniform(1.01,10)),
    X = distr_gen(sts.uniform(1, 20)),
    Y = distr_gen(sts.uniform(0,20)),
    Z = it.repeat(0),
    peak_window = it.repeat(6))

profile_generator = GenBuilder(
    classname = core.AxialCorrection,
    profile = symmetric_profile_generator,
    HL = distr_gen(sts.uniform(0.0005, 0.1)),
    SL = distr_gen(sts.uniform(0.0005,0.1)),
    N_gauss_step = it.repeat(GAUSS_STEPS))

## phases

phase_generator = GenBuilder(
    classname = core.Phase,
    structure = str_generator,
    profile = profile_generator)

# patterns

GRID = np.linspace(3.0, 90.0, 4351)
CUKA1 = [[1.540596, 1]]
CUKA12 = [[1.540596, 2/3], [1.544493, 1/3]]

pattern_generator = GenBuilder(
    classname= core.Pattern,
    waves = sample_gen([CUKA1, CUKA12]),
    phases = ([phase] for phase in phase_generator),
    bkg = bkg_generator,
    scales = distr_gen(sts.uniform(100,20000), size = 1),
    bkg_range = (sorted(ii) for
                 ii in utils.distr_gen(sts.uniform(500, 7000), size = 2)))

# running

def runner(pattern):
    return (pattern.report_params(), pattern.pattern(GRID))

import multiprocessing as mp
import pandas as pd

CHANKS = 2
CHANK_SIZE = 50
CORES = 4

for i in range(CHANKS):
    chank = it.islice(pattern_generator, CHANK_SIZE)
    pool = mp.Pool(CORES)
    with pool as p:
        results = p.map(runner, chank)

    y = pd.DataFrame( [ y for y,_ in results  ]  )
    x = pd.DataFrame( [ x for _,x in results  ]  )

    y.to_csv(f'test_y_{i}.csv')
    x.to_csv(f'test_x_{i}.csv')
    pool.close()
