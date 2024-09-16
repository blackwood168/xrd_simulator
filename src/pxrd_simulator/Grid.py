#!/usr/bin/env python3

import numpy as np
import scipy.stats as stts

class Grid:
    def __init__(self, start, end, step, peak_width):
        self.x0 = start
        self.xn = end
        self.n = round((end - start)/step) + 1
        self.grid = np.linspace(start, end, self.n)
        self.phase_grid = np.zeros(self.n)
        self.bkg_x = np.linspace(-1, 1, self.n)
        self.bkg = np.zeros(self.n)
        self.step = self.grid[1] - self.grid[0]
        self.delta = peak_width / 2.0
    def add_peak(self, th2_i, profile):
        th2 = th2_i[0]
        i   = th2_i[1]
        where = (self.grid > th2 - self.delta) & (self.grid < th2 + self.delta)
        # todo: replace with constant time index arithmetics (benchmark?)
        self.phase_grid[where] += i * profile.calc(self.grid[where], th2)
    def add_peaks(self, th2, i, profile, bin_size = None):
        if bin_size is not None:
            n_bins = round((self.xn - self.x0) / bin_size) + 1
            bins = np.linspace(self.x0, self.xn, n_bins)
            indexes = np.digitize(th2, bins)
            tt = np.bincount(indexes)
            mask = tt > 0
            new_i = np.bincount(indexes, i)[mask]
            new_th2 = np.bincount(indexes, th2)[mask] / tt[mask]
            peaks = zip(new_th2, new_i)
        else:
            peaks = zip(th2, i)
        for peak in peaks:
            self.add_peak(peak, profile)
    def calc_bkg(self, coefs):
        self.coefs = coefs
        self.bkg = np.polynomial.chebyshev.chebval(self.bkg_x, coefs)
    def calc_pattern(self, max_I, bkg_low, bkg_high, seed):
        np.random.seed(seed)
        phase_norm = self.phase_grid / max(self.phase_grid)
        if np.abs(max(self.bkg) - min(self.bkg)) < 1E-8:
            bkg_norm = self.bkg + 1
        else:
            bkg_norm = (self.bkg - min(self.bkg)) / (max(self.bkg) - min(self.bkg))
        pattern = max_I * (bkg_low + (bkg_high - bkg_low) * bkg_norm + (1 - bkg_high) * phase_norm)
        self.bkg_scaled =  (bkg_high - bkg_low) * bkg_norm * max_I + bkg_low * max_I
        scale = (bkg_high - bkg_low) * max_I / (max(self.bkg) - min(self.bkg) )
        delta = max_I * bkg_low - scale * min(self.bkg)
        new_coefs = self.coefs * scale
        new_coefs[0] += delta
        self.coefs_scaled = new_coefs
        return np.random.poisson(pattern)
