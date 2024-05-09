#!/cluster/tufts/dinglab/shared_apps/miniconda3/envs/lab/bin/python
#SBATCH --job-name=compute_energy
#SBATCH --partition=dinglab
#SBATCH --time=1:00:00
#SBATCH --array=0-22
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --open-mode=trucate
#SBATCH --output=./slurm_output/compute_energy_%A_%a.out

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import pickle
import os
import time
import argparse
import mdtraj
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="water")

args = parser.parse_args()
phase = args.phase

idx_lambda = int(os.environ['SLURM_ARRAY_TASK_ID'])

with open(f"./output/{phase}_phase/lambdas.pkl", "rb") as f:
    lambdas_list = pickle.load(f)

lambdas = lambdas_list[idx_lambda]

(elec0, vdw0), (elec1, vdw1) = lambdas
lambdas_str = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"

## deserialize the system
with open(
    f"./output/{phase}_phase/sys/{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}.xml",
    "r",
) as f:
    system = mm.XmlSerializer.deserialize(f.read())

## add barostat
if phase == "water":
    system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 298.15 * unit.kelvin))

with open(f"./output/{phase}_phase/topology.pkl", "rb") as f:
    topology = pickle.load(f)
topology = mdtraj.Topology.from_openmm(topology)

pdb = app.PDBFile(f"./output/{phase}_phase/system.pdb")

integrator = mm.LangevinMiddleIntegrator(
    298.15 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
)
kbT = 298.15 * unit.kelvin * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

platform = mm.Platform.getPlatformByName("CUDA")
simulation = app.Simulation(topology, system, integrator, platform)

## load trajectories
reduced_u = []
for lambdas in lambdas_list:
    (elec0, vdw0), (elec1, vdw1) = lambdas
    lambdas_str_traj = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"
    traj = mdtraj.load(
        f"./output/{phase}_phase/traj/{lambdas_str_traj}.dcd",
        top=topology,
    )
    
    if phase == "water":
        for xyz, unit_cell_vectors in zip(traj.xyz, traj.unitcell_vectors):
            simulation.context.setPositions(xyz)
            simulation.context.setPeriodicBoxVectors(*unit_cell_vectors)
            u = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kbT
            reduced_u.append(u)
    else:
        for xyz in traj.xyz:
            simulation.context.setPositions(xyz)
            u = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kbT
            reduced_u.append(u)

reduced_u = np.array(reduced_u)

os.makedirs(f"./output/{phase}_phase/reduced_potentials", exist_ok=True)
with open(f"./output/{phase}_phase/reduced_potentials/{lambdas_str}.pkl", "wb") as f:
    pickle.dump(reduced_u, f)