#!/usr/bin/env python3

import numpy as np
import itertools as it

from numpy.random.mtrand import dirichlet

def bragg(d, wave):
        rad = np.arcsin(wave / (2 * d))
        return np.degrees(rad) * 2

def unbragg(Th2, wave):
        theta = np.radians(Th2/2)
        return wave / 2 / np.sin(theta)

def get(thing):
        try:
                result = thing.__next__()
        except AttributeError:
                result = thing
        return result

def sample_gen(values, p = None, size = None):
    def generator():
        rng = np.random.default_rng()
        N = len(values)
        try:
                vals = np.array(values)
        except ValueError:
                vals = np.array(values, dtype='object')
        while True:
            iis = rng.choice(N, p = get(p), size = get(size))
            if (size is None):
                item = vals[iis]
            else:
                item = vals[np.array(iis)]
            yield item
    return generator()

def distr_gen(frozen_distr, size = None):
    def generator():
        while True:
            item = frozen_distr.rvs(size = get(size))
            yield item
    return generator()

class GenBuilder:
    def __init__(self, classname, **kwargs):
        self.classname = classname
        self.params = kwargs

    def next_params(self):
        return {key:value.__next__() for key, value in self.params.items()}

    def __next__(self):
            params = self.next_params()
            return self.classname.__call__(**params)
    def __iter__(self):
        return self

class CctbxStrJson:
    def __init__(self, structure):
        self.structure = structure

    def scatterer_to_dict(self, scatterer):
        d = {
            'label' : scatterer.label,
            'site' : scatterer.site,
            'u' : scatterer.u_iso,
            'occupancy' : scatterer.occupancy,
            'scattering_type' : scatterer.scattering_type,
            'fp' : scatterer.fp,
            'fpd' : scatterer.fpd
        }
        return d

    def symmetry_to_dict(symmetry):
        d = {
            'unit_cell' : symmetry.unit_cell().parameters(),}
