#!/usr/bin/env python3
import numpy as np

class PV_TCHZ:
    def __init__(self, parameters):
        self.U = parameters[0] / 1083  # Follow GSAS conventions
        self.V = parameters[1] / 1083  # Follow GSAS conventions
        self.W = parameters[2] / 1083  # Follow GSAS conventions
        self.X = parameters[3] / 100   # Follow GSAS conventions
        self.Y = parameters[4] / 100   # Follow GSAS conventions
        self.Z = parameters[5] / 100   # Follow GSAS conventions

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


    def calc(self, Th2, peak):
        wl = self.fwhmL(peak)
        wg = self.fwhmG(peak)
        n = self.n_for_tchz(wl, wg)
        return self.tchz(Th2, peak, wl, wg, n)
