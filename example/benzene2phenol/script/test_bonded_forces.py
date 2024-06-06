import openmm as mm
from openmm import XmlSerializer
import openmm.app as app
import openmm.unit as unit
import numpy as np
import xml.etree.ElementTree as ET
from sys import exit
import pickle
from atom.functions import (
    align_coordinates,
    make_graph,
    make_alchemical_system,
)

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

for f in envi_system.getForces():
    if f.__class__.__name__ == "NonbondedForce":
        break
f.setUseDispersionCorrection(False)

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

lambdas = [(0.0, 1.0), (0.0, 1.0)]

system_xml, top, coor = make_alchemical_system(
    ligs,
    [liga_top, ligb_top],
    ligs_common_atoms,
    ligs_coor,
    lambdas,
    None,
    None,
    None,
)

system = mm.XmlSerializer.deserialize(
    ET.tostring(system_xml, xml_declaration=True).decode()
)


def get_energy(system, name, coor):
    for force in system.getForces():
        if force.__class__.__name__ == name:
            force.setForceGroup(1)

    platform = mm.Platform.getPlatformByName("CUDA")
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    context = mm.Context(system, integrator, platform)
    context.setPositions(coor)
    state = context.getState(getEnergy=True, groups=set([1]))
    for force in system.getForces():
        if force.__class__.__name__ == name:
            force.setForceGroup(0)
    return state.getPotentialEnergy()



for name in ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"]:
    u = get_energy(system, name, coor).value_in_unit(
        unit.kilojoule_per_mole
    )
    u_a = get_energy(liga_system, name, liga_coor).value_in_unit(
        unit.kilojoule_per_mole
    )
    u_b = get_energy(ligb_system, name, ligb_coor).value_in_unit(
        unit.kilojoule_per_mole
    )
    assert np.isclose(u, u_a + u_b, atol=1e-5)
    print(name, u, u_a + u_b)