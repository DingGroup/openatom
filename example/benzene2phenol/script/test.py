import openmm as omm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import mdtraj
import xml.etree.ElementTree as ET
from sys import exit
import networkx as nx
from atom.functions import (
    get_graph,
    get_maximum_common_substructure,
    label_particles,
    merge_and_index_particles,
    merge_constraints,
    merge_harmonic_bonds,
    merge_harmonic_angles,
    merge_periodic_torsions,
    merge_nonbonded_forces
)
import copy

solvent_prmtop = app.AmberPrmtopFile("./structure/output/solvent.prmtop")
solvent_system = solvent_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)

iph_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop")
liga_system = iph_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)

bnz_prmtop = app.AmberPrmtopFile("./structure/output/BNZ.prmtop")
ligb_system = bnz_prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
)

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


bnz_top = mdtraj.load_mol2("./structure/output/BNZ.mol2").topology
iph_top = mdtraj.load_mol2("./structure/output/IPH.mol2").topology

mcs = get_maximum_common_substructure(iph_top, bnz_top)

liga_common_atoms = list(mcs.keys())
liga_alchem_atoms = [i for i in range(iph_top.n_atoms) if i not in liga_common_atoms]

ligb_common_atoms = [mcs[i] for i in liga_common_atoms]
ligb_alchem_atoms = [i for i in range(bnz_top.n_atoms) if i not in ligb_common_atoms]

ligs_common_atoms = [liga_common_atoms, ligb_common_atoms]
ligs_alchem_atoms = [liga_alchem_atoms, ligb_alchem_atoms]
ligs = [liga, ligb]

graphs = [get_graph(iph_top), get_graph(bnz_top)]

for lig, common_atoms, graph in zip(ligs, ligs_common_atoms, graphs):
    label_particles(lig, common_atoms, graph)    

system = ET.Element("System", solvent.attrib)
system.append(solvent.find("./PeriodicBoxVectors"))

scaling_factors = [[0.0, 0.0], [0.5, 1.0]]
merge_and_index_particles(system, solvent, ligs, ligs_common_atoms)
merge_constraints(system, solvent, ligs)
merge_harmonic_bonds(system, solvent, ligs, scaling_factors)
merge_harmonic_angles(system, solvent, ligs, scaling_factors)
merge_periodic_torsions(system, solvent, ligs, scaling_factors)
# merge_nonbonded_forces(system, solvent, ligs, scaling_factors)


tree = ET.ElementTree(system)
ET.indent(tree.getroot())
tree.write("./output/test.xml", xml_declaration=True, method="xml", encoding="utf-8")


tree = ET.ElementTree(ligs[0])
ET.indent(tree.getroot())
tree.write(
    "./output/liga_running.xml", xml_declaration=True, method="xml", encoding="utf-8"
)

tree = ET.ElementTree(ligs[1])
ET.indent(tree.getroot())
tree.write(
    "./output/ligb_running.xml", xml_declaration=True, method="xml", encoding="utf-8"
)

tree = ET.ElementTree(solvent)
ET.indent(tree.getroot())
tree.write(
    "./output/solvent_running.xml", xml_declaration=True, method="xml", encoding="utf-8"
)

# def update_particle_idx(root):
#     for ele in root.iter():
#         for n in range(1, 5):
#             if f"p{n}" in ele.attrib:
#                 current_idx = int(ele.get(f"p{n}"))
#                 new_idx = root.find("./Particles")[current_idx].get("idx")
#                 ele.set(f"p{n}", new_idx)


# for root in ligs + [solvent]:
#     update_particle_idx(root)
