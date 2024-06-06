import pickle
import numpy as np
from FastMBAR import FastMBAR
import argparse
import openmm.unit as unit
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="water")

args = parser.parse_args()
phase = args.phase

kbT = 298.15 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilocalorie_per_mole)

Fs = {}
for phase in ["vacuum", "water"]:
    with open(
        f"./output/{phase}_phase/lambdas.pkl", "rb"
    ) as f:
        lambdas_list = pickle.load(f)

    u_list = []
    for lambdas in lambdas_list:
        (elec0, vdw0), (elec1, vdw1) = lambdas
        lambdas_str = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"
        with open(
            f"./output/{phase}_phase/reduced_potentials/{lambdas_str}.pkl",
            "rb",
        ) as f:
            u = pickle.load(f)

        u_list.append(u)

    u = np.array(u_list)
    num_confs = np.array([u.shape[1]//u.shape[0] ] * u.shape[0])

    mbar = FastMBAR(u, num_confs, cuda=True, verbose=True, method="L-BFGS-B")
    Fs[phase] = mbar.F * kbT

F_water = Fs["water"]
F_vacuum = Fs["vacuum"]

dF_water = F_water[-1] - F_water[0]
dF_vacuum = F_vacuum[-1] - F_vacuum[0]

dF = dF_water - dF_vacuum