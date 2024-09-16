#!/usr/bin/env python3

import numpy as np

class AxialCorrection:
    def __init__(self, profile, HL, SL, N_gauss_step):
        self.L = 1
        self.H = HL
        self.S = SL
        self.N_gauss_step = N_gauss_step
        self.profile = profile
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

    def calc(self, Th2, peak):
        phmin = self.phi_min(peak)
        dd = np.abs(peak - phmin) / self.N_gauss_step
        N_gauss = np.ceil(dd).astype(int)
        if (N_gauss == 1):
            return self.profile.calc(Th2, peak)
        xn, wn = np.polynomial.legendre.leggauss(N_gauss)
        step = Th2[1] -Th2[0]
        deltan = (peak+phmin)/2 + (peak-phmin)*xn/2
        tmp_assy = np.zeros(len(Th2))
        arr1 = wn*self.W2(deltan, peak)/self.h(deltan, peak)/np.cos(deltan*np.pi/180)
        for dn in range(len(deltan)):
            tmp_assy += arr1[dn] * self.profile.calc(Th2, deltan[dn])
        tmp_assy = tmp_assy / np.sum(arr1)
        return(tmp_assy)
