#!/usr/bin/env python3

import numpy as np
import random
from cctbx.development import random_structure
import scipy.stats as stts

class RandomStructure:
    def __init__(self, wavelength, space_groups, elements, av_atoms_n, atom_volume):
        self.wl = wavelength
        self.sgs = space_groups
        self.elements = elements
        self.n_sgs = len(space_groups)
        self.n_atoms = av_atoms_n
        self.vol = atom_volume
    def str_gen(self, seed):
        np.random.seed(seed)
        n = stts.poisson(self.n_atoms).rvs(1)[0].item()
        elements = list(np.random.choice(self.elements, size = n))
        random.seed(seed)
        group = self.sgs[random.randrange(self.n_sgs)]
        self.structure = random_structure.xray_structure(
            elements = elements,
            space_group_symbol = group,
            volume_per_atom = self.vol,
            random_u_iso=True
        )
        return(self.structure)
    def peaks(self, th2_max):
        dmin =self.wl / 2 / np.sin(th2_max / 2 / 180 * np.pi  )
        a = self.structure.structure_factors(d_min= dmin).f_calc().sort()
        I = a.as_intensity_array().data().as_numpy_array()
        m = a.multiplicities().data().as_numpy_array()
        I *= m
        Th2 = a.two_theta(self.wl, deg = True).data().as_numpy_array()
        return Th2, I

## Example:
## rs = RandomStructure(1.540618, ["P21/c"], ["C", "N", "O", "F"], 25, 18)
## rs.str_gen(2532)
## rs.peaks(15)
##(array([ 9.89911285, 10.5843743 , 12.45903905, 12.93762182, 13.09775555,
##        13.47545917, 14.65882659]),
## array([ 4165.2820222 , 16176.83872192, 16092.04591525, 13126.03224443,
##         4113.61833053,  4846.62032893,  3150.89355266]))
