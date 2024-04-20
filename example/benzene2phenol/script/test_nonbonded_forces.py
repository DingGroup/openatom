import openmm as omm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
from atom.functions import (
    align_coordinates,
    get_graph,
    get_maximum_common_substructure,
    label_particles,
    merge_and_index_particles,
    merge_constraints,
    merge_nonbonded_forces,
    make_custom_nonbonded_force,
)

solvent_prmtop = app.AmberPrmtopFile("./structure/output/solvent.prmtop")
solvent_system = solvent_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=1.0 * unit.nanometer,
)
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

liga_solvated_coor = app.AmberInpcrdFile(
    "./structure/output/IPH_solvated.inpcrd"
).getPositions()
liga_solvated_coor = np.array(liga_solvated_coor.value_in_unit(unit.nanometer))
liga_coor = liga_solvated_coor[0 : liga_top.getNumAtoms()]

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
ligb_solvated_coor = app.AmberInpcrdFile(
    "./structure/output/BNZ_solvated.inpcrd"
).getPositions()
ligb_solvated_coor = np.array(ligb_solvated_coor.value_in_unit(unit.nanometer))
ligb_coor = ligb_solvated_coor[0 : ligb_top.getNumAtoms()]


solvent_xml = XmlSerializer.serializeSystem(solvent_system)
liga_xml = XmlSerializer.serializeSystem(liga_system)
ligb_xml = XmlSerializer.serializeSystem(ligb_system)


solvent = ET.fromstring(solvent_xml)
liga = ET.fromstring(liga_xml)
ligb = ET.fromstring(ligb_xml)


mcs = get_maximum_common_substructure(liga_top, ligb_top)
liga_common_atoms = list(mcs.keys())
liga_alchem_atoms = [
    i for i in range(liga_top.getNumAtoms()) if i not in liga_common_atoms
]
ligb_common_atoms = [mcs[i] for i in liga_common_atoms]
ligb_alchem_atoms = [
    i for i in range(ligb_top.getNumAtoms()) if i not in ligb_common_atoms
]
ligs_common_atoms = [liga_common_atoms, ligb_common_atoms]
ligs_alchem_atoms = [liga_alchem_atoms, ligb_alchem_atoms]
ligs = [liga, ligb]

ligb_coor = align_coordinates(
    liga_coor,
    ligb_coor,
    ligs_common_atoms[0],
    ligs_common_atoms[1],
)
ligb_coor[ligb_common_atoms] = liga_coor[liga_common_atoms]
ligs_coor = [liga_coor, ligb_coor]


ligb_solvated_coor[0 : ligb_coor.shape[0]] = ligb_coor
ligb_solvated_coor[ligb_coor.shape[0] :] = liga_solvated_coor[liga_coor.shape[0] :]


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


graphs = [get_graph(liga_top), get_graph(ligb_top)]
for lig, common_atoms, graph in zip(ligs, ligs_common_atoms, graphs):
    label_particles(lig, common_atoms, graph)

system_xml = ET.Element("System", solvent.attrib)
system_xml.append(solvent.find("./PeriodicBoxVectors"))


particles = merge_and_index_particles(ligs, ligs_common_atoms, solvent)
system_xml.append(particles)


n = len(system_xml.findall("./Particles/Particle"))
coor = np.zeros((n, 3))
for idx_lig, (lig, lig_coor) in enumerate(zip(ligs, ligs_coor)):
    for i, p in enumerate(lig.iterfind("./Particles/Particle")):
        if idx_lig == 0:
            j = int(p.get("idx"))
            coor[j] = lig_coor[i]
        else:
            if p.get("class") == "soft-core":
                j = int(p.get("idx"))
                coor[j] = lig_coor[i]
for i, p in enumerate(solvent.iterfind("./Particles/Particle")):
    j = int(p.get("idx"))
    coor[j] = solvent_coor[i]
coor = np.array(coor)


constraints = merge_constraints(ligs, solvent)
system_xml.append(constraints)

scaling_factors = [[0.0, 0.0], [1.0, 1.0]]
forces = ET.SubElement(system_xml, "Forces")
force = merge_nonbonded_forces(ligs, scaling_factors, solvent)
forces.append(force)

force = make_custom_nonbonded_force(ligs, scaling_factors, solvent)
forces.append(force)

tree = ET.ElementTree(system_xml)
ET.indent(tree.getroot())
tree.write(
    "./output/test_nonbonded.xml", xml_declaration=True, method="xml", encoding="utf-8"
)

system_xml_string = ET.tostring(system_xml, xml_declaration=True).decode()
system = omm.XmlSerializer.deserialize(system_xml_string)

for lig, lig_system in zip(ligs, [liga_solvated_system, ligb_solvated_system]):
    for f in lig_system.getForces():
        if f.__class__.__name__ == "NonbondedForce":
            break
    f.setUseDispersionCorrection(False)

name = "NonbondedForce"
ea = get_energy(liga_solvated_system, name, liga_solvated_coor)
eb = get_energy(ligb_solvated_system, name, ligb_solvated_coor)
esys = get_energy(system, name, coor) + get_energy(system, "CustomNonbondedForce", coor)
print("ea: ", ea)
print("eb: ", eb)
print("esys: ", esys)