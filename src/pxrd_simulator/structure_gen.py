import core
import utils

from utils import GenBuilder, distr_gen, sample_gen

import numpy as np
import scipy.stats as sts
import itertools as it

# structure
SPACE_GROUPS = ['P212121', 'P21', 'C2', 'P21212', 'C2221', 'P1'] #PDB

#SPACE_GROUPS = ['P21/c', "P-1", "P21", 'P212121', "C2/c", "Pbca"] #CSD
ELEMENTS = ["C", "N", "O", "Cl"]
N_ATOMS_LIMS = (10, 30)
ATOM_VOLUME_START_WIDTH = (14, 8)

d_high = 0.8
d_low = 1.5

n_atoms = sample_gen(range(*N_ATOMS_LIMS))

str_generator = GenBuilder(
    classname= core.CctbxStr.generate_packing,
    sg = sample_gen(SPACE_GROUPS),
    atoms = sample_gen(ELEMENTS, size = n_atoms),
    atom_volume = distr_gen( sts.uniform(*ATOM_VOLUME_START_WIDTH)),
    seed = utils.distr_gen(sts.randint(1, 2**32-1)))

# running

def runner(pattern):
    params = pattern.report_params()
    structure = pattern.structure
    a_high = structure.structure_factors(d_min= d_high).f_calc().sort()
    I_high = a_high.as_intensity_array().data().as_numpy_array()
    m_high = a_high.multiplicities().data().as_numpy_array()
    I_high *= m_high
    ind_high = np.array(list(a_high.indices()))
    a_low = structure.structure_factors(d_min= d_low).f_calc().sort()
    #I_low = a_low.as_intensity_array().data().as_numpy_array()
    #m_low = a_low.multiplicities().data().as_numpy_array()
    #I_low *= m_low
    ind_low = np.array(list(a_low.indices()))
    return {'structure_params': params, 'ind_low': ind_low, 'I_high': I_high, 'ind_high': ind_high}

import multiprocessing as mp
import pandas as pd

CHANKS = 10
CHANK_SIZE = 10000
CORES = 2

for i in range(CHANKS):
    chank = it.islice(str_generator, CHANK_SIZE)
    pool = mp.Pool(CORES)
    with pool as p:
        results = p.map(runner, chank)

    np.savez_compressed(f'csd_{d_low}_{d_high}_{i}', db = np.array(results))
    #x.to_csv(f'test_x_{i}.csv')
    pool.close()