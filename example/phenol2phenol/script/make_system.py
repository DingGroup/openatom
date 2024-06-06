import openmm as omm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
import os
import pickle
import time
from atom.functions import (
    align_coordinates,
    make_graph,
    make_alchemical_system,
)
import argparse
from dltoolbox import make_psf_from_topology

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="water")

args = parser.parse_args()
phase = args.phase


envi_prmtop = app.AmberPrmtopFile("./structure/output/solvent.prmtop")
envi_system = envi_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=0.9 * unit.nanometer,
)

envi_top = envi_prmtop.topology
envi_coor = app.AmberInpcrdFile("./structure/output/solvent.inpcrd").getPositions()
envi_coor = np.array(envi_coor.value_in_unit(unit.nanometer))


liga_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop", envi_top.getPeriodicBoxVectors())
liga_system = liga_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=0.9 * unit.nanometer,
)
liga_top = liga_prmtop.topology
liga_coor = app.AmberInpcrdFile("./structure/output/IPH.inpcrd").getPositions()
liga_coor = np.array(liga_coor.value_in_unit(unit.nanometer))

ligb_prmtop = app.AmberPrmtopFile("./structure/output/IPH.prmtop", envi_top.getPeriodicBoxVectors())
ligb_system = ligb_prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    switchDistance=0.9 * unit.nanometer,
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

mcs = {0:0, 1:1, 2:2, 3:3, 4:4, 
       5:5, 7:7, 8:8, 9:9, 
       10:10, 11:11}

liga_common_atoms = list(mcs.keys())
ligb_common_atoms = [mcs[i] for i in liga_common_atoms]
ligs_common_atoms = [liga_common_atoms, ligb_common_atoms]

graphs = [make_graph(liga_top), make_graph(ligb_top)]


ligb_coor = align_coordinates(
    liga_coor, ligb_coor, liga_common_atoms, ligb_common_atoms
)

ligs_coor = [liga_coor, ligb_coor]


start_time = time.time()
lambdas_list = [
    [(1.0, 1.0), (0.0, 0.0)],
    [(0.8, 1.0), (0.0, 0.0)],
    [(0.6, 1.0), (0.0, 0.0)],
    [(0.4, 1.0), (0.0, 0.0)],
    [(0.2, 1.0), (0.0, 0.0)],
    [(0.0, 1.0), (0.0, 0.0)],
    [(0.0, 0.95), (0.0, 0.05)],
    [(0.0, 0.9), (0.0, 0.1)],
    [(0.0, 0.8), (0.0, 0.2)],
    [(0.0, 0.7), (0.0, 0.3)],
    [(0.0, 0.6), (0.0, 0.4)],
    [(0.0, 0.5), (0.0, 0.5)],
    [(0.0, 0.4), (0.0, 0.6)],
    [(0.0, 0.3), (0.0, 0.7)],
    [(0.0, 0.2), (0.0, 0.8)],
    [(0.0, 0.1), (0.0, 0.9)],
    [(0.0, 0.05), (0.0, 0.95)],
    [(0.0, 0.0), (0.0, 1.0)],
    [(0.0, 0.0), (0.2, 1.0)],
    [(0.0, 0.0), (0.4, 1.0)],
    [(0.0, 0.0), (0.6, 1.0)],
    [(0.0, 0.0), (0.8, 1.0)],
    [(0.0, 0.0), (1.0, 1.0)],
]

os.makedirs(f"./output/{phase}_phase", exist_ok=True)
with open(f"./output/{phase}_phase/lambdas.pkl", "wb") as f:
    pickle.dump(lambdas_list, f)

for lambdas in lambdas_list:
    print(lambdas)
    if phase == "vacuum":
        envi, envi_top, envi_coor = None, None, None
        
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

    tree = ET.ElementTree(system_xml)
    ET.indent(tree.getroot())
    (elec0, vdw0), (elec1, vdw1) = lambdas
    os.makedirs(f"./output/{phase}_phase/sys", exist_ok=True)
    tree.write(
        f"./output/{phase}_phase/sys/{elec0:.2f}_{vdw0:.2f}_{elec1:.2f}_{vdw1:.2f}.xml",
        xml_declaration=True,
        method="xml",
        encoding="utf-8",
    )


    omm.app.PDBFile.writeFile(
        top, coor * 10, f"./output/{phase}_phase/system.pdb", keepIds=True
    )


    with open(f"./output/{phase}_phase/topology.pkl", "wb") as file_handle:
        pickle.dump(top, file_handle)


    make_psf_from_topology(top, f"./output/{phase}_phase/topology.psf")