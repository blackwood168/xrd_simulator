import core
import utils
from utils import GenBuilder, distr_gen, sample_gen

import numpy as np
import scipy.stats as sts
import itertools as it
import multiprocessing as mp
from multiprocessing import freeze_support
import pandas as pd

# Structure space groups for different databases/systems
#SPACE_GROUPS = ['P212121', 'P21', 'C2', 'P21212', 'C2221', 'P1']      # PDB
#SPACE_GROUPS = ['P21/c', "P-1", "P21", 'P212121', "C2/c", "Pbca"]     # CSD
#SPACE_GROUPS = ['C2221', 'P21212', 'P212121']                         # Orthorhombic
SPACE_GROUPS = ['P21', 'C2']                                           # Monoclinic

# Structure composition parameters
ELEMENTS = ["C", "N", "O", "Cl"]
N_ATOMS_LIMS = (10, 30)
ATOM_VOLUME_START_WIDTH = (14, 8)

# Resolution limits
d_high = 0.8  # High resolution limit
d_low = 1.5   # Low resolution limit

# Generate number of atoms
n_atoms = sample_gen(range(*N_ATOMS_LIMS))

# Initialize structure generator
str_generator = GenBuilder(
    classname=core.CctbxStr.generate_packing,
    sg=sample_gen(SPACE_GROUPS),
    atoms=sample_gen(ELEMENTS, size=n_atoms),
    atom_volume=distr_gen(sts.uniform(*ATOM_VOLUME_START_WIDTH)),
    seed=utils.distr_gen(sts.randint(1, 2**32-1))
)


def runner(pattern):
    """Process a single crystal structure pattern.
    
    Args:
        pattern: Crystal structure pattern object
        
    Returns:
        dict: Contains structure parameters and intensity data
    """
    params = pattern.report_params()
    structure = pattern.structure
    
    # Calculate structure factors and intensities
    a_high = structure.structure_factors(d_min=d_high).f_calc().sort()
    I_high = a_high.as_intensity_array().data().as_numpy_array()
    ind_high = np.array(list(a_high.indices()))
    
    a_low = structure.structure_factors(d_min=d_low).f_calc().sort()
    ind_low = np.array(list(a_low.indices()))
    
    return {
        'structure_params': params,
        'ind_low': ind_low,
        'I_high': I_high,
        'ind_high': ind_high
    }


if __name__ == '__main__':
    freeze_support()
    
    # Processing parameters
    CHANKS = 100
    CHANK_SIZE = 10000
    CORES = 8

    # Process structures in chunks
    for i in range(CHANKS):
        chank = it.islice(str_generator, CHANK_SIZE)
        pool = mp.Pool(CORES)
        
        with pool as p:
            results = p.map(runner, chank)

        # Save results
        np.savez_compressed(
            f'/home/arthur/xray_ml/pxrd_sim/src/pxrd_simulator/clin_test_100k/clin{d_low}_{d_high}_{i}',
            db=np.array(results)
        )
        pool.close()
