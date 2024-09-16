import numpy as np
import itertools as it
import logging
from cctbx.development import random_structure
import iotbx.cif
import random
import pkg_resources

# grid = np.linspace(start, stop, N)
class Pattern:
    def __init__(self, waves, phases, bkg, scales,  bkg_range):
        self.phases = list(zip(phases, scales))
        self.bkg = bkg
        self.bkg_range = bkg_range
        self.bkg.scale_to_range(bkg_range)
        self.waves = waves

    def pattern_clean(self, grid):
        pattern = self.bkg.bkg(grid)
        for (phase, scale) in self.phases:
            p = phase.calc_phase(grid, self.waves)
            pp = p / np.max(p) * scale
            pattern = pattern + pp
        return np.array(pattern, dtype=np.float64)

    def pattern(self, grid):
        pattern = self.pattern_clean(grid)
        return np.random.poisson(pattern)

    def report_params(self):
        phases = {f'phase{i}_{key}': value
                  for i, (phase, _) in enumerate(self.phases)
                  for key, value in phase.report_params().items()}
        scales = {f'phase{i}_max': scale
                  for i, (_, scale) in enumerate(self.phases)}
        bkg = {f'bkg_{k}': v for k, v in self.bkg.report_params().items()}
        bkg['bkg_min'] = self.bkg_range[0]
        bkg['bkg_max'] = self.bkg_range[1]
        y = {**scales, **phases, **bkg}
        return y


class Phase:
    def __init__(self, structure, profile, lp_angle=0, bin_size = 0.04):
        self.structure = structure
        self.profile = profile
        self.lp_angle = lp_angle
        self.bin_size = bin_size

    def report_params(self):
        return {**self.structure.report_params(),
                **self.profile.report_params()}

    def bragg(self, d, wave):
        rad = np.arcsin(wave / (2 * d))
        return np.degrees(rad) * 2

    def unbragg(self, Th2, wave):
        theta = np.radians(Th2/2)
        return wave / 2 / np.sin(theta)

    def peaks_2th(self, peaks, waves):
        ds = peaks[:,0]
        Is = peaks[:,1]
        result = np.empty((0,2))
        for wl, wI in waves:
            th2_peaks = np.column_stack((
                self.bragg(ds, wl),
                Is * wI ))
            result = np.vstack((result, th2_peaks))
        order = result[:,0].argsort()
        return result[order]

    def lp(self, th2_peaks):
        if self.lp_angle is None :
            return th2_peaks
        A = np.cos( np.radians(self.lp_angle) * 2 )**2
        th2 = np.radians(th2_peaks[:,0])
        correction = (1 + A * np.cos(th2)**2) / (1 + A) / np.sin(th2)
        result = np.copy(th2_peaks)
        result[:,1] = result[:,1] * correction
        return result

    def bin(self, th2_peaks):
        MIN_NONZERO = 1e-14
        th2 = th2_peaks[:,0]
        th2_min = min(th2)
        th2_max = max(th2)
        i = np.fmax(th2_peaks[:,1], MIN_NONZERO)
        n_bins = round((th2_max - th2_min) / self.bin_size) + 1
        bins = np.linspace(th2_min, th2_max, n_bins)
        indexes = np.digitize(th2, bins)
        tt = np.bincount(indexes)
        mask = tt > 0
        new_i = np.bincount(indexes, i)[mask]
        # new_th2 = np.bincount(indexes, th2)[mask] / tt[mask]
        new_th2 = np.bincount(indexes, th2*i)[mask] / new_i
        return np.column_stack((new_th2, new_i))

    def dmin(self, grid, waves):
        min_wave = np.min([wl for wl,_ in waves])
        max_Th2 = np.max(grid)
        return self.unbragg(max_Th2, min_wave)

    def prepare_peaks(self, grid, waves, dmin = None):
        if dmin is None:
            dmin = self.dmin(grid, waves)
        peaks = self.peaks_2th(self.structure.peaks(dmin), waves)
        peaks = self.lp(peaks)
        if self.bin_size is not None:
            peaks = self.bin(peaks)
        return peaks

    def calc_phase(self, grid, waves, dmin = None):
        peaks = self.prepare_peaks(grid, waves, dmin)
        phase = np.zeros(len(grid))
        for peak in peaks:
            position = peak[0]
            intensity = peak[1]
            start = np.searchsorted(grid, position - self.profile.window / 2, "left")
            stop = np.searchsorted(grid, position + self.profile.window / 2, "left")
            phase[start:stop] += self.profile.profile(position,
                                                      grid[start:stop]) * intensity
        return phase

class Structure:
    def peaks(self):
        raise NotImplementedError(
            'peaks method should be defined in %s' % (self.__class__.__name__))

    def report_params(self):
        raise NotImplementedError(
            'report_params method should be defined in %s' % (self.__class__.__name__))

    def set_dmax(self, dmax):
        self.dmax = dmax


class CctbxStr(Structure):
    def __init__(self, structure):
        self.seed = 0
        self.structure = structure

    def peaks(self, dmin):
        a = self.structure.structure_factors(d_min= dmin).f_calc().sort()
        I = a.as_intensity_array().data().as_numpy_array()
        m = a.multiplicities().data().as_numpy_array()
        I *= m
        d = a.d_spacings().data().as_numpy_array()
        result = np.array([d, I]).T
        return result

    def report_params(self):
        stru = self.structure
        symmetry = stru.crystal_symmetry()
        cell = symmetry.unit_cell().parameters()
        volume = symmetry.unit_cell().volume()
        group = symmetry.space_group_info().symbol_and_number()
        result = {k:v for k,v in
                  zip(("cell_a", "cell_b", "cell_c",
                       "cell_alpha", "cell_beta", "cell_gamma"),
                      cell)}
        result["cell_volume"] = volume
        result["group"] = group
        result["structure_n_atoms"] = len(stru.scatterers())
        return result

    @classmethod
    def from_cif(cls, cif_file, block_number=1):
        strucs = iotbx.cif.reader(cif_file).\
        build_crystal_structures()
        _ , struc = next(it.islice(strucs.items(), 1))
        return CctbxStr(struc)

    @classmethod
    def generate_packing(self, sg, atoms, atom_volume, seed):
        np.random.seed(seed)
        random.seed(seed)
        structure = random_structure.xray_structure(
            elements = list(atoms),
            space_group_symbol = sg,
            volume_per_atom = atom_volume,
            random_u_iso = True
        )
        result = CctbxStr(structure)
        result.seed = seed
        return result


class Background:
    def __init__(self, domain=[-1, 1], inner_grid_size=200):
        self.domain_min = domain[0]
        self.domain_max = domain[1]
        self.domain_range = domain[1] - domain[0]
        self.inner_grid = np.linspace(domain[0], domain[1], inner_grid_size)

    def min(self):
        return np.min(self.domain_bkg(self.inner_grid))

    def max(self):
        return np.max(self.domain_bkg(self.inner_grid))

    def map_grid_to_domain(self, grid):
        scale = self.domain_range / (grid[-1] - grid[0])
        return scale * grid - scale * grid[0] + self.domain_min

    def bkg(self, grid):
        domain_grid = self.map_grid_to_domain(grid)
        return self.domain_bkg(domain_grid)

    def scale_to_range(self, bkg_range):
        current_range = self.max() - self.min()
        new_min = bkg_range[0]
        new_max = bkg_range[1]
        scale = (new_max - new_min) / current_range
        self.scale(scale)
        shift = new_min - self.min()
        self.shift(shift)

    def domain_bkg(self, grid):
        raise NotImplementedError(
            'domain_bkg method should be defined in %s' % (self.__class__.__name__))

    def scale(self, scale):
        raise NotImplementedError(
            'scale method should be defined in %s' % (self.__class__.__name__))

    def shift(self, scale):
        raise NotImplementedError(
            'shift method should be defined in %s' % (self.__class__.__name__))

    def report_params(self):
        raise NotImplementedError(
            'report_params method should be defined in %s' % (self.__class__.__name__))


class ChebyshevBkg(Background):
    def __init__(self, coefs, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        new_coefs = [v if i <= order else 0 for i, v in enumerate(coefs)]
        self.coefs = np.array(new_coefs)

    def domain_bkg(self, grid):
        return np.polynomial.chebyshev.chebval(grid, self.coefs)

    def shift(self, x):
        self.coefs[0] = self.coefs[0] + x

    def scale(self, x):
        self.coefs = self.coefs * x

    def report_params(self):
        return {f'ch{i}': v for i, v in enumerate(self.coefs)}


class Profile:
    def __init__(self, peak_window):
        self.window = peak_window

    def profile(self, peak_position, subgrid):
        raise NotImplementedError(
            'profile method should be defined in %s' % (self.__class__.__name__))

    def report_params(self):
        raise NotImplementedError(
            'report_params method should be defined in %s' % (self.__class__.__name__))


class PV_TCHZ(Profile):
    def __init__(self, U, V, W, X, Y, Z, **kwargs):
        super().__init__(**kwargs)
        self.U = U / 1083  # Follow GSAS conventions
        self.V = V / 1083  # Follow GSAS conventions
        self.W = W / 1083  # Follow GSAS conventions
        self.X = X / 100   # Follow GSAS conventions
        self.Y = Y / 100   # Follow GSAS conventions
        self.Z = Z / 100   # Follow GSAS conventions

    def fwhmL(self, peak):
        peak = peak / 180 * np.pi
        return (self.X * np.tan(peak/2) + self.Y/np.cos(peak/2))

    def fwhmG(self, peak):
        peak = peak / 180 * np.pi
        return np.sqrt(self.U * np.tan(peak/2) ** 2 +
                       self.V * np.tan(peak/2) +
                       self.W +
                       self.Z / np.cos(peak/2) ** 2)

    def lorenz(self, Th2, peak, l):
        return (2 / np.pi / l) / (1 + 4 * (Th2 - peak)**2 / l**2)

    def gauss(self, Th2, peak, g):
        return (2 * (np.log(2)/np.pi) ** 0.5 / g) * np.exp(-4 * np.log(2) * (Th2 - peak)**2 / g**2)

    def n_for_tchz(self, l, g):
        G = g ** 5 + 2.69269*g ** 4 * l + 2.42843 * g ** 3 * l ** 2 + 4.47163 * g ** 2 * l ** 3
        G += 0.07842 * g * l ** 4 + l ** 5
        G = l / (G ** 0.2)
        n = 1.36603 * G - 0.47719 * G ** 2 + 0.11116 * G ** 3
        return n

    def tchz(self, Th2, peak, l, g, n):
        return n* self.lorenz(Th2, peak, l) + (1 - n)* self.gauss(Th2, peak, g)

    def profile(self, peak, Th2):
        wl = self.fwhmL(peak)
        wg = self.fwhmG(peak)
        n = self.n_for_tchz(wl, wg)
        return self.tchz(Th2, peak, wl, wg, n)

    def report_params(self):
        return self.__dict__


class AxialCorrection(Profile):
    with pkg_resources.resource_stream(__name__, '../../data/leggauss.npy') as f:
        LEGGAUSS = np.load(f, allow_pickle=True)

    def leggauss(self, N):
        if N < len(AxialCorrection.LEGGAUSS) :
            result = AxialCorrection.LEGGAUSS[N]
        else:
            logging.warning(f'Large legendre polinom degree: {N}')
            result = np.polynomial.legendre.leggauss(N)
        return result

    def __init__(self, profile, HL, SL, N_gauss_step):
        super().__init__(profile.window)
        self.L = 1
        self.H = HL
        self.S = SL
        self.N_gauss_step = N_gauss_step
        self.sym = profile
#        self.xn, self.wn = np.polynomial.legendre.leggauss(N_gauss)

    def h(self, phi, peak):
        return self.L*np.sqrt(np.cos(phi*np.pi/180)**2/np.cos(peak*np.pi/180)**2 - 1)

    def phi_min(self, peak):
        a = np.cos(peak*np.pi/180) * np.sqrt( ((self.H+self.S)/self.L)**2 + 1 )
        if a > 1 :
            return 0
        else:
            return 180/np.pi*np.arccos( a )
    def phi_infl(self, peak):
        a = np.cos(peak*np.pi/180)*np.sqrt( ((self.H-self.S)/self.L)**2 + 1 )
        if a > 1 :
            return 0
        else:
            return 180/np.pi*np.arccos(a)

    def W2(self, phis, peak):
        result = np.zeros(len(phis))
        cond1 = (self.phi_min(peak) <= phis) & (phis <= self.phi_infl(peak))
        result[cond1] = self.H + self.S - self.h(phis[cond1], peak)
        cond2 = (phis > self.phi_infl(peak)) & (phis <= peak)
        result[cond2] = 2 * min(self.H, self.S)
        return result

    def profile(self, peak, Th2):
        phmin = self.phi_min(peak)
        dd = np.abs(peak - phmin) / self.N_gauss_step
        N_gauss = np.ceil(dd).astype(int)
        logging.debug(f'Peak position is {peak}, min phi is {phmin}, integrating with {N_gauss} steps.')
        if (N_gauss == 1):
            return self.sym.profile(peak, Th2)
        xn, wn = self.leggauss(N_gauss)
        deltan = (peak+phmin)/2 + (peak-phmin)*xn/2
        tmp_assy = np.zeros(len(Th2))
        arr1 = wn*self.W2(deltan, peak)/self.h(deltan, peak)/np.cos(deltan*np.pi/180)
        for dn in range(len(deltan)):
            if deltan[dn] > peak - (self.window / 2 * 1.2) :
                tmp_assy += arr1[dn] * self.sym.profile(deltan[dn], Th2)
        tmp_assy = tmp_assy / np.sum(arr1)
        return(tmp_assy)

    def report_params(self):
        return {**self.sym.report_params(), 'HL': self.H,
                'SL': self.S, 'NGaussSteps': self.N_gauss_step}


def script_main():

    print("Script successfully executed")
