import unittest
import copy
import numpy as np
import matplotlib.pyplot as plt
from src.pxrd_simulator.core import (ChebyshevBkg, Profile,
                                     Structure, Phase,
                                     Background, Pattern,
                                     PV_TCHZ, AxialCorrection,
                                     CctbxStr)
from scipy.stats import norm
import importlib.resources as pkg_res
import tests

# Wavelengthes
CUKA1 = [[1.540596, 1]]
CUKA12 = [[1.540596, 2/3], [1.544493, 1/3]]

# peaks
ONE_PEAK = [[20, 1]]
TWO_PEAKS = [[20, 1], [2 ,1]]

# grid
GRID = np.linspace(3, 90, int((90 - 3) / 0.01) + 1)

# Bruker D2 profile (from Kaduk)
D2_TCHZ = {'U': 1.376, 'V': -0.928, 'W': 2.640, 'X': 2.410, 'Y': 0.850, 'Z': 0 }
D2_AXIAL = {'SL': 0.02951, 'HL':  0.02951, 'N_gauss_step': 0.01}

# quartz structure cif
QUARTZ_CIF = str(pkg_res.files(tests) / "quartz.cif")

class GaussProfile(Profile):
    def __init__(self, window, sigma):
        super().__init__(window)
        self.sigma = sigma
    def profile(self, peak_position, subgrid):
        return norm.pdf(subgrid, loc = peak_position, scale = self.sigma)
    def report_params(self):
        return {'PeakSigma':self.sigma}

class DummyStructure(Structure):
    def __init__(self, peaks):
        self.p = np.array(peaks)
    def peaks(self, dmin):
        return self.p
    def report_params(self):
        return {'Npeaks':len(self.p)}

class LinearBkg(Background):
    '''ax+b background, defined on [-1,1]'''
    def __init__(self, a, b, **kwds):
        super().__init__(**kwds)
        self.a = a
        self.b = b
    def domain_bkg(self, grid):
        return self.a * grid + self.b
    def shift(self, x):
        self.b = self.b + x
    def scale(self, x):
        self.a = self.a * x
        self.b = self.b * x
    def report_params(self):
        return {'a':self.a, 'b':self.b}

class TestBackground(unittest.TestCase):
    def setUp(self):
        self.bkg = LinearBkg(2, -5)
        self.ch_bkg = ChebyshevBkg([3,2,1], 2)
        self.grid = np.linspace(10, 20, 11)
    def test_range(self):
        self.assertEqual(self.bkg.min(), -7.0)
        self.assertEqual(self.bkg.max(),  -3.0)
    def test_rescale(self):
        bkg = copy.deepcopy(self.bkg)
        bkg.scale_to_range((0.0, 4.0))
        self.assertEqual(bkg.min(), 0.0)
        self.assertEqual(bkg.max(), 4.0)
    def test_rescale1(self):
        bkg = copy.deepcopy(self.bkg)
        bkg.scale_to_range((0.0, 4.0))
        self.assertEqual(bkg.a, 2.0)
    def test_points(self):
        points = self.bkg.bkg(self.grid)
        self.assertEqual(points[0], -7.0)
        self.assertEqual(points[-1], -3.0)
    def test_cheb_rescale(self):
        bkg = copy.deepcopy(self.ch_bkg)
        bkg.scale_to_range((10.0,20.0))
        self.assertEqual(bkg.min(), 10.0)
        self.assertEqual(bkg.max(), 20.0)
    def test_cheb_linear(self):
        ch = ChebyshevBkg([5,4,3,2,1], 1)
        bkg = LinearBkg(a = 4, b = 5.0)
        np.testing.assert_allclose(ch.bkg(self.grid),
                                   bkg.bkg(self.grid))
        bkg.scale_to_range((10.0,20.0))
        ch.scale_to_range((10.0,20.0))
        np.testing.assert_allclose(ch.bkg(self.grid),
                                   bkg.bkg(self.grid))


class TestPhase(unittest.TestCase):
    def setUp(self):
        self.waves = [(1.54, 1)]
        self.grid = np.linspace(0,7, 701)
        self.profile = GaussProfile(10, 0.5)
        self.one_peak_str = DummyStructure([[22, 1]])
        self.two_peaks_str = DummyStructure([[29.5, 1], [14.7,1]])
    def test_calc_phase_one_peak(self):
        ph = Phase(self.one_peak_str, self.profile, lp_angle=None)
        x1 = ph.calc_phase(self.grid, self.waves)
        th2 =ph.bragg(22, self.waves[0][0])
        x2 = self.profile.profile(th2, self.grid)
        # plt.plot(x1),
        # plt.plot(x2)
        # plt.show()
        self.assertTrue(np.array_equal(x1, x2))
    def test_calc_phase_two_peaks(self):
        ph = Phase(self.two_peaks_str, self.profile, lp_angle=None)
        x1 = ph.calc_phase(self.grid, self.waves)
        x1 = ph.calc_phase(self.grid, self.waves)
        th2_1 =ph.bragg(29.5, self.waves[0][0])
        th2_2 =ph.bragg(14.7, self.waves[0][0])
        x2 = self.profile.profile(th2_1, self.grid) + \
            self.profile.profile(th2_2, self.grid)
        # plt.plot(x1),
        # plt.plot(x2)
        # plt.show()
        self.assertTrue(np.array_equal(x1, x2))


class TestPattern(unittest.TestCase):
    def setUp(self):
        self.grid = np.linspace(10, 40, 701)
        self.profile1 = GaussProfile(1, 0.2)
        self.profile2 = GaussProfile(3, 1)
        self.one_peak_str = DummyStructure([[4.44, 5]])
        self.two_peaks_str = DummyStructure([[29.5, 1], [14.7,1]])
        self.bkg = LinearBkg(2, -5)
        self.phase1 = Phase(self.one_peak_str, self.profile2)
        self.phase2 = Phase(self.two_peaks_str, self.profile1)
        self.pattern = Pattern([[1.54, 1]],
                               [self.phase1, self.phase2], self.bkg,
                               (300,300), (10, 20))
    def test_report(self):
        params = self.pattern.report_params()
        self.assertEqual(params['bkg_min'], 10)
        self.assertEqual(params['phase0_max'], 300)
        self.assertEqual(params['phase0_Npeaks'], 1)
        self.assertEqual(params['phase1_Npeaks'], 2)
        self.assertEqual(params['phase0_PeakSigma'], 1)
        self.assertEqual(params['phase1_PeakSigma'], 0.2)
    # def test_plot(self):
    #     points  = self.pattern.pattern(self.grid)
    #     plt.plot(self.grid, points)
    #     plt.show()

class TestAxialPVTCHZ(unittest.TestCase):
    def setUp(self):
        self.grid = GRID
        self.tchz = PV_TCHZ(peak_window = 10, **D2_TCHZ)
        self.axial = AxialCorrection(self.tchz, **D2_AXIAL)
        self.str = DummyStructure(TWO_PEAKS)
        self.bkg = LinearBkg(2, -5)
    def test_axial_peak(self):
        grid = np.linspace(3,5, 10000)
        axial = self.axial.profile(4, grid)
        tchz = self.tchz.profile(4, grid)
        self.assertEqual(len(tchz), len(grid))
        self.assertEqual(len(axial), len(grid))
        # plt.plot(grid, tchz)
        # plt.plot(grid, axial)
        # plt.show()
    def test_axial_plot(self):
        sym_phase = Phase(self.str, self.tchz)
        asym_phase = Phase(self.str, self.axial)
        asym_pat = Pattern(CUKA12, [asym_phase] , self.bkg, [1000], (1,2))
        sym_pat = Pattern(CUKA12, [sym_phase] , self.bkg, [1000], (1,2))
        asym_points = asym_pat.pattern_clean(self.grid)
        sym_points = sym_pat.pattern_clean(self.grid)
        self.assertEqual(len(sym_points), len(self.grid))
        self.assertEqual(len(asym_points), len(self.grid))
        # plt.plot(self.grid, sym_points)
        # plt.plot(self.grid, asym_points)
        # plt.show()


class TestCctbx(unittest.TestCase):
    def setUp(self):
        self.stru = CctbxStr.from_cif(QUARTZ_CIF)
    def test_peaks(self):
           peaks = self.stru.peaks(2)
           self.assertEqual(len(peaks), 7)
