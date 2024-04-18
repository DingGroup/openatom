import openmm as omm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import mdtraj
from atom.functions import (
    get_maximum_common_substructure,
    get_nonbonded_force,
    get_nonbonded_parameters,
)
from sys import exit

iph_solvated_prmtop = app.AmberPrmtopFile("./structure/output/IPH_solvated.prmtop")
lig_0_solvated_system = iph_solvated_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)

bnz_prmtop = app.AmberPrmtopFile("./structure/output/BNZ.prmtop")
lig_1_system = bnz_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)

sys_xml = XmlSerializer.serializeSystem(lig_1_system)


exit()

bnz_top = mdtraj.load_mol2("./structure/output/BNZ.mol2").topology
iph_top = mdtraj.load_mol2("./structure/output/IPH.mol2").topology
mcs = get_maximum_common_substructure(iph_top, bnz_top)

lig_0_common_atoms = list(mcs.keys())
lig_0_alchem_atoms = [i for i in range(iph_top.n_atoms) if i not in lig_0_common_atoms]

lig_1_common_atoms = [mcs[i] for i in lig_0_common_atoms]
lig_1_alchem_atoms = [i for i in range(bnz_top.n_atoms) if i not in lig_1_common_atoms]

common_atoms_nbf_params_0 = {"charge": [], "sigma": [], "epsilon": []}
common_atoms_nbf_params_1 = {"charge": [], "sigma": [], "epsilon": []}

nbf_0 = get_nonbonded_force(lig_0_solvated_system)
nbf_1 = get_nonbonded_force(lig_1_system)

common_atoms_nbf_params_0 = get_nonbonded_parameters(nbf_0, lig_0_common_atoms)
common_atoms_nbf_params_1 = get_nonbonded_parameters(nbf_1, lig_1_common_atoms)

lig_0_alchem_atoms_nbf_params = get_nonbonded_parameters(nbf_0, lig_0_alchem_atoms)
lig_1_alchem_atoms_nbf_params = get_nonbonded_parameters(nbf_1, lig_1_alchem_atoms)


## add lig_1 alchemical atoms to the system
lig_1_alchem_atoms_map = {}
N = lig_0_solvated_system.getNumParticles()
for i in lig_1_alchem_atoms:
    lig_0_solvated_system.addParticle(lig_1_system.getParticleMass(i))
    lig_1_alchem_atoms_map[i] = N
    N += 1

base_charge_0 = []
base_charge_1 = []
for i in range(len(lig_0_common_atoms)):
    base_charge_0.append(
        [lig_0_common_atoms[i], common_atoms_nbf_params_0["charge"][i]]
    )
    base_charge_1.append(
        [lig_0_common_atoms[i], common_atoms_nbf_params_1["charge"][i]]
    )

for i in range(len(lig_0_alchem_atoms)):
    base_charge_0.append(
        [lig_0_alchem_atoms[i], lig_0_alchem_atoms_nbf_params["charge"][i]]
    )
    base_charge_1.append([lig_0_alchem_atoms[i], 0.0])

for i in range(len(lig_1_alchem_atoms)):
    base_charge_0.append([lig_1_alchem_atoms_map[lig_1_alchem_atoms[i]], 0.0])
    base_charge_1.append(
        [
            lig_1_alchem_atoms_map[lig_1_alchem_atoms[i]],
            lig_1_alchem_atoms_nbf_params["charge"][i],
        ]
    )

exit()

## add lig_1 alchemical atoms to the nonbonded force
for i in lig_1_alchem_atoms:
    nbf_0.addParticle(0.0, 1.0, 0.0)
