"""
Command line interface for pxrd_simulator package
"""

import argparse
from ipydex import IPS, activate_ips_on_exception
import scipy.stats as stts
import numpy as np
from . AxialCorrection import AxialCorrection
from . PVTCHZ import PV_TCHZ
from . Grid import Grid
from . RandomStructure import RandomStructure


activate_ips_on_exception()

rs = RandomStructure(1.540618,
                     ["P-1", "P21", "P21/c", "C2/c", "P212121", "Pbca"],
                     ["C", "N", "O", "F"], 25, 18)

N = 10
def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog = 'pxrd_simuator',
        description = 'Simulate powder patterns with randomized phase, profile, and background')

    parser.add_argument(
        'out', metavar = 'OUT',
        help = "Output pxrd data and parameters to CSV files OUT.X and OUT.Y")           # positional argument
    parser.add_argument('-s', '--samples', metavar = 'N',
                        type=int, default = 100,
                        help='Number of samples to produce')

    args = parser.parse_args()
    print("Hey!\n")
    print(args)

    N = args.samples

    Cheb_N = 13
    Cheb_zero_prob = 0.4

    #  Create random parameters (Y)


    seed = stts.randint(1, 10e7).rvs(N).reshape(N,1)
    Imax = stts.uniform(500, 20000).rvs(N).reshape(N,1)
    ab = stts.uniform(0,0.5).rvs(2 * N).reshape((N,2))
    bkg = stts.norm(0,1).rvs(Cheb_N * N).reshape(N, Cheb_N) * \
    stts.binom(1, 1- Cheb_zero_prob).rvs(Cheb_N * N).reshape(N, Cheb_N)

    gu = stts.uniform(1, 5).rvs(N).reshape(N,1)
    gv =  - stts.uniform(0, 1).rvs(N).reshape(N,1)
    gw = stts.uniform(1.01,10).rvs(N).reshape(N,1)
    gx = stts.uniform(1, 20).rvs(N).reshape(N,1)
    gy = stts.uniform(0,20).rvs(N).reshape(N,1)
    gz = stts.uniform(0, 0).rvs(N).reshape(N,1)
    sl_plus_hl  = stts.uniform(0.0005, 0.1).rvs(N).reshape(N,1)
    sl_minus_hl = stts.norm(0,0.002).rvs(N).reshape(N,1)


    sl = np.abs((sl_plus_hl + sl_minus_hl)/2)
    hl = np.abs((sl_plus_hl - sl_minus_hl)/2)

    Y = np.hstack((
        seed, gu, gv,gw,gx,gy, gz,
        hl, sl, Imax, ab, bkg
    ))

    outY = "{}.Y".format(args.out)
    np.savetxt(outY, Y, delimiter=",")

    print("Just wrote randomized parameters to {}".format(outY))

    def calc_pattern(params):
        local_seed = round(params[0])
        cogliotti = params[1:7]
        axials = params[7:9]
        Intensity = params[9]
        bkg_low = params[10]
        bkg_high = params[11]
        bkg_coefs = params[12:]
        pv = PV_TCHZ(cogliotti)
        ac = AxialCorrection(pv, axials[0], axials[1], 0.01)
        rs.str_gen(local_seed)
        pTh2, pI = rs.peaks(90.0)
        gr = Grid(3.0,90.0, 0.02, 6.0)
        gr.add_peaks(pTh2, pI, ac, 0.08)
        gr.calc_bkg(bkg_coefs)
        pp = gr.calc_pattern(Intensity, bkg_low, bkg_high, local_seed)
        cc = gr.coefs_scaled
        return np.concatenate((cc, pp))

    result = np.apply_along_axis(calc_pattern,1, Y)

    bkg = result[:,:Cheb_N]


    outBkg = "{}.Bkg".format(args.out)
    np.savetxt(outBkg, bkg, delimiter=",")

    raw = result[:,Cheb_N:]
    outX = "{}.X".format(args.out)
    np.savetxt(outX, raw, delimiter=",")
