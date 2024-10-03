import openmm as mm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
from openatom.merge_systems import (
    make_graph,
    make_alchemical_system,
    align_coordinates,
)
import pickle
from sys import exit

envi_prmtop = app.AmberPrmtopFile("./structure/output/solvent.prmtop")
envi_system = envi_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)

envi_top = envi_prmtop.topology
envi_coor = app.AmberInpcrdFile("./structure/output/solvent.inpcrd").getPositions()
envi_coor = np.array(envi_coor.value_in_unit(unit.nanometer))

long_range_correction = True

for f in envi_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(long_range_correction)

liga_prmtop = app.AmberPrmtopFile("./structure/output/BNZ.prmtop")
liga_system = liga_prmtop.createSystem(
    nonbondedMethod=app.CutoffNonPeriodic,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
liga_top = liga_prmtop.topology
liga_coor = app.AmberInpcrdFile("./structure/output/BNZ.inpcrd").getPositions()
liga_coor = np.array(liga_coor.value_in_unit(unit.nanometer))

ligb_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop")
ligb_system = ligb_prmtop.createSystem(
    nonbondedMethod=app.CutoffNonPeriodic,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
ligb_top = ligb_prmtop.topology
ligb_coor = app.AmberInpcrdFile("./structure/output/IPH.inpcrd").getPositions()
ligb_coor = np.array(ligb_coor.value_in_unit(unit.nanometer))


envi_xml = XmlSerializer.serializeSystem(envi_system)
liga_xml = XmlSerializer.serializeSystem(liga_system)
ligb_xml = XmlSerializer.serializeSystem(ligb_system)


envi = ET.fromstring(envi_xml)
liga = ET.fromstring(liga_xml)
ligb = ET.fromstring(ligb_xml)
ligs = [liga, ligb]

with open("./output/mcs.pkl", "rb") as f:
    mcs = pickle.load(f)

liga_common_atoms = list(mcs.keys())
ligb_common_atoms = [mcs[i] for i in liga_common_atoms]
ligs_common_atoms = [liga_common_atoms, ligb_common_atoms]

graphs = [make_graph(liga_top), make_graph(ligb_top)]
ligb_coor = align_coordinates(
    liga_coor, ligb_coor, liga_common_atoms, ligb_common_atoms
)
ligs_coor = [liga_coor, ligb_coor]

lambdas = [(1.0, 1.0), (0.0, 0.0)]
#lambdas = [(0.0, 0.0), (1.0, 1.0)]

system_xml, top, coor = make_alchemical_system(
    ligs,
    [liga_top, ligb_top],
    ligs_common_atoms,
    ligs_coor,
    lambdas,
    envi,
    envi_top,
    envi_coor,
)

system = mm.XmlSerializer.deserialize(
    ET.tostring(system_xml, xml_declaration=True).decode()
)


liga_ref_prmtop = app.AmberPrmtopFile("./structure/output/BNZ_solvated.prmtop")
liga_ref_system = liga_ref_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
for f in liga_ref_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(long_range_correction)

liga_ref_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
liga_ref_coor = np.concatenate([liga_coor, envi_coor])


ligb_ref_prmtop = app.AmberPrmtopFile("./structure/output/IPH_solvated.prmtop")
ligb_ref_system = ligb_ref_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
for f in ligb_ref_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(long_range_correction)

ligb_ref_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
ligb_ref_coor = np.concatenate([ligb_coor, envi_coor])

def get_energy(system, name, coor):
    if name is not None:
        for force in system.getForces():
            if force.__class__.__name__ == name:
                force.setForceGroup(1)

    platform = mm.Platform.getPlatformByName("Reference")
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    context = mm.Context(system, integrator, platform)
    context.setPositions(coor)

    if name is not None:
        state = context.getState(getEnergy=True, groups=set([1]))
    else:
        state = context.getState(getEnergy=True)

    if name is not None:
        for force in system.getForces():
            if force.__class__.__name__ == name:
                force.setForceGroup(0)
    return state.getPotentialEnergy()

name = None
u = get_energy(system, name, coor).value_in_unit(unit.kilojoule_per_mole)
ua = get_energy(liga_ref_system, name, liga_ref_coor).value_in_unit(unit.kilojoule_per_mole)
ub = get_energy(ligb_ref_system, name, ligb_ref_coor).value_in_unit(unit.kilojoule_per_mole)



exit()



name = 'NonbondedForce'
ua = get_energy(liga_ref_system, name, liga_ref_coor).value_in_unit(unit.kilojoule_per_mole)
ub = get_energy(ligb_ref_system, name, ligb_ref_coor).value_in_unit(unit.kilojoule_per_mole)
u = get_energy(system, 'NonbondedForce', coor).value_in_unit(unit.kilojoule_per_mole) + get_energy(system, 'CustomNonbondedForce', coor).value_in_unit(unit.kilojoule_per_mole)

print(u, ua, ub)
exit()

for name in ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"]:
    u = get_energy(system, name, coor).value_in_unit(
        unit.kilojoule_per_mole
    )
    u_a = get_energy(liga_ref_system, name, liga_ref_coor).value_in_unit(
        unit.kilojoule_per_mole
    )
    u_b = get_energy(ligb_ref_system, name, ligb_ref_coor).value_in_unit(
        unit.kilojoule_per_mole
    )
    assert np.isclose(u, u_a + u_b, atol=1e-5)
    print(name, u, u_a + u_b)

exit()


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

    platform = omm.Platform.getPlatformByName("CUDA")
    integrator = omm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    context = omm.Context(system, integrator, platform)
    context.setPositions(coor)

    #omm.LocalEnergyMinimizer.minimize(context)

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