import openmm as omm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
from sys import exit
from atom.functions import (
    align_coordinates,
    get_graph,
    get_maximum_common_substructure,
    label_particles,
    merge_and_index_particles,
    merge_constraints,
    merge_harmonic_bonds,
    merge_harmonic_angles,
    merge_periodic_torsions,
)


solvent_prmtop = app.AmberPrmtopFile("./structure/output/solvent.prmtop")
solvent_system = solvent_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)
solvent_coor = app.AmberInpcrdFile("./structure/output/solvent.inpcrd").getPositions()
solvent_coor = np.array(solvent_coor.value_in_unit(unit.nanometer))


liga_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop")
liga_system = liga_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)
liga_top = liga_prmtop.topology
liga_coor = app.AmberInpcrdFile("./structure/output/IPH.inpcrd").getPositions()
liga_coor = np.array(liga_coor.value_in_unit(unit.nanometer))


ligb_prmtop = app.AmberPrmtopFile("./structure/output/BNZ.prmtop")
ligb_system = ligb_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)
ligb_top = ligb_prmtop.topology
ligb_coor = app.AmberInpcrdFile("./structure/output/BNZ.inpcrd").getPositions()
ligb_coor = np.array(ligb_coor.value_in_unit(unit.nanometer))

solvent_xml = XmlSerializer.serializeSystem(solvent_system)
liga_xml = XmlSerializer.serializeSystem(liga_system)
ligb_xml = XmlSerializer.serializeSystem(ligb_system)

with open("./output/solvent.xml", "w") as f:
    f.write(solvent_xml)

with open("./output/liga.xml", "w") as f:
    f.write(liga_xml)

with open("./output/ligb.xml", "w") as f:
    f.write(ligb_xml)


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
    liga_coor, ligb_coor, ligs_common_atoms[0], ligs_common_atoms[1]
)
ligb_coor[ligb_common_atoms] = liga_coor[liga_common_atoms]
ligs_coor = [liga_coor, ligb_coor]


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

scaling_factors = [[0.0, 1.0], [0.0, 1.0]]
forces = ET.SubElement(system_xml, "Forces")

force = merge_harmonic_bonds(ligs, scaling_factors, solvent)
forces.append(force)

force = merge_harmonic_angles(ligs, scaling_factors, solvent)
forces.append(force)

force = merge_periodic_torsions(ligs, scaling_factors, solvent)
forces.append(force)

system_xml_string = ET.tostring(system_xml, xml_declaration=True).decode()
system = omm.XmlSerializer.deserialize(system_xml_string)

for name in ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"]:
    print("name: ", name)
    esys = get_energy(system, name, coor)
    ea = get_energy(liga_system, name, liga_coor)
    eb = get_energy(ligb_system, name, ligb_coor)
    es = get_energy(solvent_system, name, solvent_coor)
    print(esys - ea - eb - es)

tree = ET.ElementTree(system_xml)
ET.indent(tree.getroot())
tree.write("./output/test.xml", xml_declaration=True, method="xml", encoding="utf-8")

exit()
