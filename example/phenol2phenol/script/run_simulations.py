#!/cluster/tufts/dinglab/shared_apps/miniconda3/envs/lab/bin/python
#SBATCH --job-name=run_simulation
#SBATCH --partition=dinglab
#SBATCH --time=24:00:00
#SBATCH --array=0-22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --open-mode=trucate
#SBATCH --output=./slurm_output/run_simulation_%A_%a.out

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import pickle
import os
import time
import argparse
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="water")

args = parser.parse_args()
phase = args.phase


idx_lambda = int(os.environ["SLURM_ARRAY_TASK_ID"])

with open(f"./output/{phase}_phase/lambdas.pkl", "rb") as f:
    lambdas_list = pickle.load(f)
lambdas = lambdas_list[idx_lambda]
(elec0, vdw0), (elec1, vdw1) = lambdas
lambdas_str = f"{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}"
print(f"Running simulation for lambdas {lambdas_str}", flush=True)

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

pdb = app.PDBFile(f"./output/{phase}_phase/system.pdb")

integrator = mm.LangevinMiddleIntegrator(
    298.15 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
)
platform = mm.Platform.getPlatformByName("CUDA")
simulation = app.Simulation(topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)


print("Minimizing energy", flush=True)
simulation.minimizeEnergy(tolerance=1.0)

simulation.integrator.setStepSize(0.001 * unit.picoseconds)
for T in np.linspace(50, 298.15, 10):
    integrator.setTemperature(T * unit.kelvin)
    simulation.context.setVelocitiesToTemperature(T * unit.kelvin)
    print(f"Equilibrating at {T} K", flush=True)
    simulation.step(10_000)

simulation.integrator.setStepSize(0.002 * unit.picoseconds)

os.makedirs(f"./output/{phase}_phase/traj", exist_ok=True)
simulation.reporters.append(
    app.DCDReporter(f"./output/{phase}_phase/traj/{lambdas_str}.dcd", 5_000)
)

print("Running simulation")
start_time = time.time()
simulation.step(5_000_000)

simulation.saveCheckpoint(f"./output/{phase}_phase/traj/{lambdas_str}.chk")
print(f"Simulation finished in {time.time() - start_time:.2f} seconds")
