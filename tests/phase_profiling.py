#!/usr/bin/env python3

import src.pxrd_simulator.core as core
import src.pxrd_simulator.utils as utils

import numpy as np
import scipy.stats as sts
import itertools as it
import pickle
import cProfile
import pstats

with open("phase_profiler_data.pcl", 'rb') as file :
    profiler_data = pickle.load(file)


GAUSS_STEPS = 0.01
GRID = np.linspace(3.0, 90.0, 4351)
CUKA12 = [[1.540596, 2/3], [1.544493, 1/3]]

tchz = [core.PV_TCHZ(u, v, w, x, y, z,
                     peak_window = 6.0) for
                     u, v, w, x, y, z in
                     profiler_data['sp']]

profile_generator = [core.AxialCorrection(pr, hl, sl, GAUSS_STEPS) for
                     pr, (hl, sl) in
                     zip(tchz, profiler_data['ap'])]


phases = [core.Phase(stru, prof) for
                   stru, prof in zip(profiler_data['stru'], profile_generator)]


ob = cProfile.Profile()

ob.enable()
for phase in phases[0:10]:
    a = phase.calc_phase(GRID, CUKA12)
ob.disable()

stats = pstats.Stats(ob)
stats.dump_stats("initial.dat")
##   gprof2dot -f pstats initial.dat -z core:59:calc_phase  --node-label=total-time-percentage --node-label=self-time-percentage  --node-label=total-time --node-label=self-time | dot -Tpdf -o initial.initial.pdf
##

# Monkey patch np.leggauss

LEG = [()] + [ np.polynomial.legendre.leggauss(i) for i in range(1, 1001)  ]

_old_leg = np.polynomial.legendre.leggauss

DEGS = []

def _leg(i):
    DEGS.append(i)
    if i <= 1000 :
        return LEG[i]
    else:
        print(f'Large legendre polynomial degree: {i} !!')
        return _old_leg(i)


np.polynomial.legendre.leggauss = _leg



ob = cProfile.Profile()
ob.enable()
for phase in phases[0:100]:
    a = phase.calc_phase(GRID, CUKA12)
ob.disable()

stats = pstats.Stats(ob)
stats.dump_stats("leggauss_fix.dat")
##   gprof2dot -f pstats leggauss_fix.dat -z core:59:calc_phase  --node-label=total-time-percentage --node-label=self-time-percentage  --node-label=total-time --node-label=self-time | dot -Tpdf -o leggauss_fix.pdf
##

# wrap Str for binning the peaks
#

class Bin:
    def __init__(self, structure, bin_size, grid):
        self.structure = structure
        self.bin_size = bin_size
        self.grid = grid
        self.min = min(grid)
        self.max = max(grid)

    def bin(self, peaks):
        th2 = [x for x, _ in peaks ]
        i = [y for _, y in peaks]
        if self.bin_size is not None:
            n_bins = round((self.max - self.min) / self.bin_size) + 1
            bins = np.linspace(self.min, self.max, n_bins)
            indexes = np.digitize(th2, bins)
            tt = np.bincount(indexes)
            mask = tt > 0
            new_i = np.bincount(indexes, i)[mask]
            new_th2 = np.bincount(indexes, th2)[mask] / tt[mask]
            new_peaks = zip(new_th2, new_i)
        else:
            new_peaks = peaks
        return list(new_peaks)

    def peaks(self, dmin):
        return self.bin( self.structure.peaks(dmin) )


phases = [core.Phase(Bin(stru, 0.02, GRID), prof) for
                   stru, prof in zip(profiler_data['stru'], profile_generator)]



ob = cProfile.Profile()

ob.enable()
for phase in phases:
    a = phase.calc_phase(GRID, CUKA12)
ob.disable()

stats = pstats.Stats(ob)
stats.dump_stats("binning_leggauss_100.dat")
##   gprof2dot -f pstats binning_leggauss.dat -z core:59:calc_phase  --node-label=total-time-percentage --node-label=self-time-percentage  --node-label=total-time --node-label=self-time | dot -Tpdf -o binning_leggauss.pdf
##
##

# gauss & lorenz from scipy

# def gauss_sts(self, Th2, peak, g):
#         return sts.norm.pdf(Th2, peak, g * core.PV_TCHZ.G_fwhm)
# core.PV_TCHZ.G_fwhm = 1 / 2 / np.sqrt(2 * np.log(2))
# core.PV_TCHZ.gauss = gauss_sts

# tchz = [core.PV_TCHZ(u, v, w, x, y, z,
#                      peak_window = 6.0) for
#                      u, v, w, x, y, z in
#                      profiler_data['sp']]

# profile_generator = [core.AxialCorrection(pr, hl, sl, GAUSS_STEPS) for
#                      pr, (hl, sl) in
#                      zip(tchz, profiler_data['ap'])]


# phases = [core.Phase(Bin(stru, 0.02, GRID), prof) for
#                    stru, prof in zip(profiler_data['stru'], profile_generator)]

# ob = cProfile.Profile()

# ob.enable()
# for phase in phases:
#     a = phase.calc_phase(GRID, CUKA12)
# ob.disable()

# stats = pstats.Stats(ob)
# stats.dump_stats("binning_leggauss_gauss_100.dat")
# ##   gprof2dot -f pstats binning_leggauss_gauss.dat -z core:59:calc_phase  --node-label=total-time-percentage --node-label=self-time-percentage  --node-label=total-time --node-label=self-time | dot -Tpdf -o binning_leggauss_gauss_100.pdf
# ##
# ## WILDLY INEFFICIENT!!!

# some peak calculation limits


def new_profile(self, peak, Th2):
    phmin = self.phi_min(peak)
    dd = np.abs(peak - phmin) / self.N_gauss_step
    N_gauss = np.ceil(dd).astype(int)
    if (N_gauss == 1):
        return self.sym.profile(peak, Th2)
    xn, wn = np.polynomial.legendre.leggauss(N_gauss) ## slow! should be tabulated
    deltan = (peak+phmin)/2 + (peak-phmin)*xn/2
    tmp_assy = np.zeros(len(Th2))
    arr1 = wn*self.W2(deltan, peak)/self.h(deltan, peak)/np.cos(deltan*np.pi/180)
    for dn in range(len(deltan)):
        if deltan[dn] > peak - (self.window / 2 * 1.2) :
            tmp_assy += arr1[dn] * self.sym.profile(deltan[dn], Th2)
    tmp_assy = tmp_assy / np.sum(arr1)
    return(tmp_assy)

core.AxialCorrection.profile = new_profile

tchz = [core.PV_TCHZ(u, v, w, x, y, z,
                     peak_window = 6.0) for
                     u, v, w, x, y, z in
                     profiler_data['sp']]

profile_generator = [core.AxialCorrection(pr, hl, sl, GAUSS_STEPS) for
                     pr, (hl, sl) in
                     zip(tchz, profiler_data['ap'])]

phases = [core.Phase(Bin(stru, 0.02, GRID), prof) for
                   stru, prof in zip(profiler_data['stru'], profile_generator)]



ob = cProfile.Profile()

ob.enable()
for phase in phases:
    a = phase.calc_phase(GRID, CUKA12)
ob.disable()

stats = pstats.Stats(ob)
stats.dump_stats("binning_leggauss_deltan_limit_100.dat")
# ##   gprof2dot -f pstats binning_leggauss_deltan_limit.dat -z core:59:calc_phase  --node-label=total-time-percentage --node-label=self-time-percentage  --node-label=total-time --node-label=self-time | dot -Tpdf -o binning_leggauss_deltan_limit_100.pdf
