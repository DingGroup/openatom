import openmm as omm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
from atom.functions import (
    make_graph,
    compute_mcs,
    make_alchemical_system,
)

solvent_prmtop = app.AmberPrmtopFile("./structure/output/solvent.prmtop")
solvent_system = solvent_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
solvent_top = solvent_prmtop.topology
solvent_coor = app.AmberInpcrdFile("./structure/output/solvent.inpcrd").getPositions()
solvent_coor = np.array(solvent_coor.value_in_unit(unit.nanometer))

for f in solvent_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(False)


liga_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop")
liga_system = liga_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
liga_top = liga_prmtop.topology


ligb_prmtop = app.AmberPrmtopFile("./structure/output/BNZ.prmtop")
ligb_system = ligb_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
ligb_top = ligb_prmtop.topology

liga_solvated_prmtop = app.AmberPrmtopFile("./structure/output/IPH_solvated.prmtop")
liga_solvated_system = liga_solvated_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
for f in liga_solvated_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(False)

liga_solvated_coor = app.AmberInpcrdFile(
    "./structure/output/IPH_solvated.inpcrd"
).getPositions()
liga_solvated_coor = np.array(liga_solvated_coor.value_in_unit(unit.nanometer))


ligb_solvated_prmtop = app.AmberPrmtopFile("./structure/output/BNZ_solvated.prmtop")
ligb_solvated_system = ligb_solvated_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
ligb_solvated_system.setDefaultPeriodicBoxVectors(
    *liga_solvated_system.getDefaultPeriodicBoxVectors()
)

for f in ligb_solvated_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(False)

ligb_solvated_coor = app.AmberInpcrdFile(
    "./structure/output/BNZ_solvated.inpcrd"
).getPositions()
ligb_solvated_coor = np.array(ligb_solvated_coor.value_in_unit(unit.nanometer))


solvent_xml = XmlSerializer.serializeSystem(solvent_system)
liga_xml = XmlSerializer.serializeSystem(liga_system)
ligb_xml = XmlSerializer.serializeSystem(ligb_system)


solvent = ET.fromstring(solvent_xml)
liga = ET.fromstring(liga_xml)
ligb = ET.fromstring(ligb_xml)
ligs = [liga, ligb]

ligs_top = [liga_top, ligb_top]


mcs = compute_mcs(liga_top, ligb_top)
liga_common_atoms = list(mcs.keys())
ligb_common_atoms = [mcs[i] for i in liga_common_atoms]
ligs_common_atoms = [liga_common_atoms, ligb_common_atoms]

graphs = [make_graph(liga_top), make_graph(ligb_top)]

# for lig, common_atoms, graph in zip(ligs, ligs_common_atoms, graphs):
#     label_particles(lig, common_atoms, graph)
# particles, top = merge_and_index_particles(ligs, ligs_top, ligs_common_atoms, solvent, solvent_top)
# exit()

liga_coor = liga_solvated_coor[0 : liga_top.getNumAtoms()]
ligb_coor = ligb_solvated_coor[0 : ligb_top.getNumAtoms()]

ligs_coor = [liga_coor, ligb_coor]

scaling_factors = [[0.0, 0.0], [1.0, 1.0]]

system_xml, top, coor = make_alchemical_system(
    ligs,
    ligs_top,
    ligs_common_atoms,
    ligs_coor,
    scaling_factors,
    solvent,
    solvent_top,
    solvent_coor,
)

ligb_solvated_coor[0 : ligb_coor.shape[0]] = ligs_coor[1]
ligb_solvated_coor[ligb_coor.shape[0] :] = liga_solvated_coor[liga_coor.shape[0] :]


system_xml_string = ET.tostring(system_xml, xml_declaration=True).decode()
system = omm.XmlSerializer.deserialize(system_xml_string)


def get_energy(system, name, coor):
    for force in system.getForces():
        if force.__class__.__name__ == name:
            force.setForceGroup(1)

    platform = omm.Platform.getPlatformByName("Reference")
    integrator = omm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    context = omm.Context(system, integrator, platform)
    context.setPositions(coor)
    state = context.getState(getEnergy=True, groups=set([1]))

    for force in system.getForces():
        if force.__class__.__name__ == name:
            force.setForceGroup(0)

    return state.getPotentialEnergy()


ea = get_energy(liga_solvated_system, "NonbondedForce", liga_solvated_coor)
eb = get_energy(ligb_solvated_system, "NonbondedForce", ligb_solvated_coor)
esys = get_energy(system, "NonbondedForce", coor) + get_energy(
    system, "CustomNonbondedForce", coor
)
print("ea: ", ea)
print("eb: ", eb)
print("esys: ", esys)

tree = ET.ElementTree(system_xml)
ET.indent(tree.getroot())
tree.write(
    "./output/test_nonbonded.xml", xml_declaration=True, method="xml", encoding="utf-8"
)