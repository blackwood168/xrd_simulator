import core
import utils
from utils import GenBuilder, distr_gen, sample_gen

import numpy as np
import scipy.stats as sts
import itertools as it
import multiprocessing as mp
from multiprocessing import freeze_support
import pandas as pd
from cctbx import crystal, miller, uctbx, sgtbx, xray
from cctbx.array_family import flex

# Structure space groups for different databases/systems
#SPACE_GROUPS = ['P212121', 'P21', 'C2', 'P21212', 'C2221', 'P1'] #PDB
#SPACE_GROUPS = ['P21/c', "P-1", "P21", 'P212121', "C2/c", "Pbca"] #CSD
#SPACE_GROUPS = ['C2221', 'P21212', 'P212121'] #orthoromb
SPACE_GROUPS = ['C2']#, 'P21'] #monoclinic

# Structure composition parameters
ELEMENTS = ["C", "N", "O", "Cl"]
#N_ATOMS_LIMS = (10, 30)
#ATOM_VOLUME_START_WIDTH = (14, 8)
N_ATOMS_LIMS = (10, 20)
ATOM_VOLUME_START_WIDTH = (14, 1)

# Resolution limits
d_high = 0.8  # High resolution limit
d_low = 1.6   # Low resolution limit

# Generate number of atoms
n_atoms = sample_gen(range(*N_ATOMS_LIMS))

# Define Miller index ranges
h_range_low, k_range_low, l_range_low = (-10, 10), (0, 10), (0, 10)
#h_range_high, k_range_high, l_range_high = (-13, 12), (0, 17), (0, 22)
h_range_high, k_range_high, l_range_high = (-15, 15), (0, 20), (0, 25)

# Generate all possible Miller indices
all_indices_low = [(h, k, l) for h in range(h_range_low[0], h_range_low[1]+1)
                           for k in range(k_range_low[0], k_range_low[1]+1)
                           for l in range(l_range_low[0], l_range_low[1]+1)]

all_indices_high = [(h, k, l) for h in range(h_range_high[0], h_range_high[1]+1)
                            for k in range(k_range_high[0], k_range_high[1]+1)
                            for l in range(l_range_high[0], l_range_high[1]+1)]

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
        dict: Contains Patterson maps, structure parameters and intensity data
    """
    params = pattern.report_params()
    structure = pattern.structure
    
    # Calculate structure factors
    a_high = structure.structure_factors(d_min=d_high).f_calc().sort().as_intensity_array()
    #print(max(list(a_high.indices())))
    a_low = structure.structure_factors(d_min=d_low).f_calc().sort().as_intensity_array()
    #print(max(list(a_low.indices())))
    
    # Process low resolution data
    existing_low = list(a_low.indices())
    existing_a_low = list(a_low.data())
    #print(existing_a_low)
    for idx in all_indices_low:
        if idx not in existing_low:
            existing_low.append(idx)
            existing_a_low.append(0.)
            
    new_low = miller.set(
        crystal_symmetry=a_low.crystal_symmetry(),
        indices=flex.miller_index(existing_low),
        anomalous_flag=a_low.anomalous_flag()
    )
    #a_low = miller.array(new_low, data = flex.double(existing_a_low))
    
    # Process high resolution data
    existing_high = list(a_high.indices())
    existing_a_high = list(a_high.data())
    
    for idx in all_indices_high:
        if idx not in existing_high:
            existing_high.append(idx)
            existing_a_high.append(0.0)
            
    new_high = miller.set(
        crystal_symmetry=a_high.crystal_symmetry(),
        indices=flex.miller_index(existing_high),
        anomalous_flag=a_high.anomalous_flag()
    )
    #a_high = miller.array(new_high, data = flex.double(existing_a_high))
    
    # Calculate Patterson maps
    patt_high = a_high.patterson_map(resolution_factor=1/2, d_min=d_high, max_prime=5, sharpening=True)
    patt_high._real_map_accessed = False
    
    patt_low = a_low.patterson_map(resolution_factor=1/2, d_min=d_low, max_prime=5, sharpening=True)
    patt_low._real_map_accessed = False
    
    # Convert to numpy arrays
    patt_low = patt_low.real_map_unpadded().as_double().as_numpy_array()
    patt_high = patt_high.real_map_unpadded().as_double().as_numpy_array()
    
    print(params,'\n',a_high.data().size(), patt_high.shape, a_low.data().size(), patt_low.shape,'\n','*'*50)
    
    # Prepare intensity and index data
    I_high = a_high.as_intensity_array().data().as_numpy_array()
    ind_high = np.array(list(a_high.indices()))
    a_low = structure.structure_factors(d_min=d_low).f_calc().sort()
    ind_low = np.array(list(a_low.indices()))
    
    return {
        'patt_low': patt_low,
        'patt_high': patt_high,
        'structure_params': params,
        'ind_low': ind_low,
        'I_high': I_high,
        'ind_high': ind_high
    }


if __name__ == '__main__':
    freeze_support()
    CHANKS = 10
    CHANK_SIZE = 10000
    CORES = 1

    for i in range(CHANKS):
        chank = it.islice(str_generator, CHANK_SIZE)
        pool = mp.Pool(CORES)
        
        with pool as p:
            results = p.map(runner, chank)
            
        #np.savez_compressed(f'clin{d_low}_{d_high}_{i}', db = np.array(results))
        np.savez_compressed(f'gg_{i}', db=np.array(results))
        pool.close()