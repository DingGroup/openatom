# For ABFE calculation

from typing import List, Tuple, Optional
from numpy import ndarray
import openmm
import openmm.app
from openmm import unit
import xml.etree.ElementTree as ET
import copy
import numpy as np
from collections import defaultdict


def make_abs_alchemical_system(
    lig: ET.Element,
    lig_top: openmm.app.Topology,
    lig_coords: ndarray,
    lambdas: List[Tuple[float, float]],
    environment: ET.Element,
    env_top: openmm.app.Topology,
    env_coords: ndarray,
) -> Tuple[ET.Element, openmm.app.Topology, np.ndarray]:
    """for ABFE calculation
    
    Args:
        lig (ET.Element): ligand XML element, from the ligand XML 
        lig_top (openmm.app.Topology): ligand OpenMM topology
        lig_coords (ndarray): ligand coordinates in np.array unit: nanometers
        lambdas (List[Tuple[float, float]]): a list of lambda values for each window first element is lambda_coul and second element is lambda_vdw
        environment (ET.Element): environment XML element, from the environment XML
        env_top (openmm.app.Topology): environment OpenMM topology
        env_coords (ndarray): Environment coordinates in np.array unit: nanometers
    
    Returns:
        Tuple[ET.Element, openmm.app.Topology, np.ndarray]: system, topology, coordinates for ABFE calculation
    """
    if environment is not None:
        system = ET.Element("System", environment.attrib)
        system.append(environment.find("./PeriodicBoxVectors"))
    else:
        system = ET.Element("System", lig.attrib)
        system.append(lig.find("./PeriodicBoxVectors"))

    particles, topology = _merge_particles_and_topology(
        lig, lig_top, environment, env_top
    )

    system.append(particles)

    constraints = _merge_constraints(lig, environment)
    system.append(constraints)

    forces = _merge_forces(lig, lambdas, environment)
    system.append(forces)

    n = len(system.findall("./Particles/Particle"))
    coor = np.zeros((n, 3))
    for i, p in enumerate(lig.findall("./Particles/Particle")):
        idx = p.get("idx")
        if idx is not None:
            j = int(idx)
            coor[j] = lig_coords[i]

    if environment is not None:
        for i, p in enumerate(environment.findall("./Particles/Particle")):
            j = p.get("idx")
            if j is not None:
                j = int(j)
                coor[j] = env_coords[i]

    coor = np.array(coor)

    return system, topology, coor


def _merge_particles_and_topology(
    lig: ET.Element,
    lig_top: openmm.app.Topology,
    env: Optional[ET.Element] = None,
    env_top: Optional[openmm.app.Topology] = None,
) -> Tuple[ET.Element, openmm.app.Topology]:
    """Merge ligand and environment particles and topologies
    
    Args:
        lig: XML element of the ligand
        lig_top: OpenMM topology of the ligand
        env: Optional, XML element of the environment
        env_top: Optional, OpenMM topology of the environment
    
    Returns:
        Tuple[ET.Element, openmm.app.Topology]: Merged particles and topology
    """
    particles = ET.Element("Particles")
    topology = openmm.app.Topology()
    lig_atoms = list(lig_top.atoms())

    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)

    p_idx = 0
    atom_names = set()

    for atom in lig.iterfind("./Particles/Particle"):
        ET.SubElement(
            particles, "Particle", {"mass": atom.get("mass")}
        )

        lig.find("./Particles")[p_idx].set("idx", str(p_idx))

        atom = lig_atoms[p_idx]
        topology.addAtom(atom.name, atom.element, residue)
        atom_names.add(atom.name)
        p_idx += 1


    if env is not None:
        chain = topology.addChain()
        residue = topology.addResidue("ENV", chain)

        for chain in env_top.chains():
            new_chain = topology.addChain(id=chain.id)
            for residue in chain.residues():
                new_residue = topology.addResidue(residue.name, new_chain)
                for atom in residue.atoms():
                    i = int(atom.index)
                    p = env.find("./Particles")[i]
                    particles.append(copy.deepcopy(p))
                    p.set("idx", str(p_idx))

                    p_idx += 1

                    topology.addAtom(atom.name, atom.element, new_residue)


    top_atoms = list(topology.atoms())
    for bond in lig_top.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        i = int(lig.find("./Particles")[i].get("idx"))
        j = int(lig.find("./Particles")[j].get("idx"))
        topology.addBond(top_atoms[i], top_atoms[j])

    if env_top is not None:
        for bond in env_top.bonds():
            i, j = bond.atom1.index, bond.atom2.index
            i = int(env.find("./Particles")[i].get("idx"))
            j = int(env.find("./Particles")[j].get("idx"))
            topology.addBond(top_atoms[i], top_atoms[j])


    if env_top is not None:
        topology.setPeriodicBoxVectors(env_top.getPeriodicBoxVectors())

    if lig_top is not None:
        topology.setPeriodicBoxVectors(lig_top.getPeriodicBoxVectors())

    return particles, topology


def _get_idx(root: ET.Element, idxs: List[int]) -> List[int]:
    idx_list = []
    for i in idxs:
        particle = root.find("./Particles")[i]
        if particle is not None:
            idx_list.append(int(particle.get("idx")))
        else:
            raise ValueError(f"Particle at index {i} not found in the XML structure.")
    return idx_list


def _merge_constraints(ligand: ET.Element, environment: ET.Element = None) -> ET.Element:
    constraints = ET.Element("Constraints")

    for c in ligand.iterfind("./Constraints/Constraint"):
        i1, i2 = int(c.get("p1")), int(c.get("p2"))
        j1, j2 = _get_idx(ligand, [i1, i2])
        ET.SubElement(
            constraints,
            "Constraint",
            {"d": c.get("d"), "p1": str(j1), "p2": str(j2)}
        )

    if environment is not None:
        for c in environment.iterfind("./Constraints/Constraint"):
            i1, i2 = int(c.get("p1")), int(c.get("p2"))
            j1, j2 = _get_idx(environment, [i1, i2])
            ET.SubElement(
                constraints,
                "Constraint",
                {"d": c.get("d"), "p1": str(j1), "p2": str(j2)}
            )

    return constraints


def _merge_forces(lig: ET.Element, lambdas: Tuple[float, float], env: ET.Element) -> ET.Element:
    forces = ET.Element("Forces")
    force = _merge_harmonic_bond_forces(lig, env)
    forces.append(force)
    force = _merge_harmonic_angle_forces(lig, env)
    forces.append(force)
    force = _merge_periodic_torsion_forces(lig, env)
    forces.append(force)

    # For nonbonded forces, we turn off the vdw interactions between ligand and environment and scale the charges of the ligand
    force = _merge_nonbonded_forces(lig, lambdas, env)
    forces.append(force)

    # for Gvdw inside the ligand
    force = _make_ligand_vdw_forces(lig)
    forces.append(force)

    # add back the Gcoul inside the ligand
    force = _make_ligand_coul_forces(lig, lambdas)
    forces.append(force)

    # softcore forces between ligand and environment
    force = _make_custom_forces(lig, lambdas, env)
    forces.append(force)


    force = _make_cmmotion_remover()
    forces.append(force)

    return forces


def _merge_harmonic_bond_forces(lig: ET.Element, env: ET.Element = None) -> ET.Element:
    force = ET.Element(
        "Force",
        {
            "forceGroup": "0",
            "name": "HarmonicBondForce",
            "type": "HarmonicBondForce",
            "usesPeriodic": "0",
            "version": "2",
        },
    )
    bonds = ET.SubElement(force, "Bonds")

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "HarmonicBondForce":
            for b in f.iterfind("./Bonds/Bond"):
                i1, i2 = int(b.get("p1")), int(b.get("p2"))
                j1, j2 = _get_idx(lig, [i1, i2])
                j1, j2 = str(j1), str(j2)
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "d": b.get("d"),
                        "k": b.get("k"),
                        "p1": str(j1),
                        "p2": str(j2),
                    },
                )

    if env is not None:
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "HarmonicBondForce":
                for b in f.iterfind("./Bonds/Bond"):
                    i1, i2 = int(b.get("p1")), int(b.get("p2"))
                    j1, j2 = _get_idx(env, [i1, i2])
                    ET.SubElement(
                        bonds,
                        "Bond",
                        {
                            "d": b.get("d"),
                            "k": b.get("k"),
                            "p1": str(j1),
                            "p2": str(j2),
                        },
                    )
    return force


def _merge_harmonic_angle_forces(lig: ET.Element, env: ET.Element = None) -> ET.Element:
    force = ET.Element(
        "Force",
        {
            "forceGroup": "0",
            "name": "HarmonicAngleForce",
            "type": "HarmonicAngleForce",
            "usesPeriodic": "0",
            "version": "2",
        },
    )
    angles = ET.SubElement(force, "Angles")

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "HarmonicAngleForce":
            for a in f.iterfind("./Angles/Angle"):
                i1, i2, i3 = int(a.get("p1")), int(
                    a.get("p2")), int(a.get("p3"))
                j1, j2, j3 = _get_idx(lig, [i1, i2, i3])
                ET.SubElement(
                    angles,
                    "Angle",
                    {'a': a.get('a'),
                        "k": str(float(a.get("k"))),
                        "p1": str(j1),
                        "p2": str(j2),
                        "p3": str(j3),
                     },
                )

    if env is not None:
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "HarmonicAngleForce":
                for a in f.iterfind("./Angles/Angle"):
                    i1, i2, i3 = int(a.get("p1")), int(
                        a.get("p2")), int(a.get("p3"))
                    j1, j2, j3 = _get_idx(env, [i1, i2, i3])
                    ET.SubElement(
                        angles,
                        "Angle",
                        {'a': a.get('a'),
                            "k": a.get("k"),
                            "p1": str(j1),
                            "p2": str(j2),
                            "p3": str(j3),
                         },
                    )
    return force


def _merge_periodic_torsion_forces(lig: ET.Element, env: ET.Element = None) -> ET.Element:
    force = ET.Element(
        "Force",
        {
            "forceGroup": "0",
            "name": "PeriodicTorsionForce",
            "type": "PeriodicTorsionForce",
            "usesPeriodic": "0",
            "version": "2",
        },
    )
    torsions = ET.SubElement(force, "Torsions")

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "PeriodicTorsionForce":
            for t in f.iterfind("./Torsions/Torsion"):
                i1, i2, i3, i4 = int(t.get("p1")), int(
                    t.get("p2")), int(t.get("p3")), int(t.get("p4"))
                j1, j2, j3, j4 = _get_idx(lig, [i1, i2, i3, i4])

                ET.SubElement(
                    torsions,
                    "Torsion",
                    {
                        "k": str(float(t.get("k"))),
                        "p1": str(j1),
                        "p2": str(j2),
                        "p3": str(j3),
                        "p4": str(j4),
                        'periodicity': t.get('periodicity'),
                        "phase": t.get("phase"),
                    },
                )

    if env is not None:
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "PeriodicTorsionForce":
                for t in f.iterfind("./Torsions/Torsion"):
                    i1, i2, i3, i4 = int(t.get("p1")), int(
                        t.get("p2")), int(t.get("p3")), int(t.get("p4"))
                    j1, j2, j3, j4 = _get_idx(env, [i1, i2, i3, i4])
                    ET.SubElement(
                        torsions,
                        "Torsion",
                        {
                            "k": t.get("k"),
                            "p1": str(j1),
                            "p2": str(j2),
                            "p3": str(j3),
                            "p4": str(j4),
                            'periodicity': t.get('periodicity'),
                            "phase": t.get("phase"),
                        },
                    )
    return force


def _merge_nonbonded_forces(
    lig: ET.Element,
    lambdas: Tuple[float, float],
    env: ET.Element = None
) -> ET.Element:
    if env is not None:
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                attrib = f.attrib
    else:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                attrib = f.attrib
    
    lambda_coul, lambda_vdw = lambdas[0], lambdas[1]
    force = ET.Element("Force", attrib)
    ET.SubElement(force, "GlobalParameters")
    ET.SubElement(force, "ParticleOffsets")
    ET.SubElement(force, "ExceptionOffsets")
    particles = ET.SubElement(force, "Particles")

    for p in lig.iterfind("./Particles/Particle"):
        ET.SubElement(
            particles, "Particle", {"eps": "0.0", "q": "0.0", "sig": "0.0"}
        )

    if env is not None:
        for p in env.iterfind("./Particles/Particle"):
            ET.SubElement(
                particles, "Particle", {"eps": "0.0", "q": "0.0", "sig": "0.0"}
            )

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            for i, p in enumerate(f.iterfind("./Particles/Particle")):
                j = int(lig.find("./Particles")[i].get("idx"))
                q = float(p.get("q")) * lambda_coul
                sig = float(p.get("sig"))
                eps = float(p.get("eps")) * lambda_vdw

                particles[j].set("eps", str(0))
                particles[j].set("q", str(q))
                particles[j].set("sig", str(sig))

    if env is not None:
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                for i, p in enumerate(f.iterfind("./Particles/Particle")):
                    j = int(env.find("./Particles")[i].get("idx"))
                    particles[j].set("eps", p.get("eps"))
                    particles[j].set("sig", p.get("sig"))
                    particles[j].set("q", p.get("q"))

    exceptions = ET.SubElement(force, "Exceptions")
    exception_dict = defaultdict(lambda: {"eps": 0, "q": 0, "sig": 0})

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            for e in f.iterfind("./Exceptions/Exception"):
                i1, i2 = int(e.get("p1")), int(e.get("p2"))
                j1, j2 = _get_idx(lig, [i1, i2])
                j1, j2 = sorted([j1, j2])
                current_eps = float(e.get("eps"))
                current_q = float(e.get("q"))
                current_sig = float(e.get("sig"))
                exception_dict[(j1, j2)]["eps"] = '0'
                exception_dict[(j1, j2)]["q"] = current_q * lambda_coul
                exception_dict[(j1, j2)]["sig"] = current_sig

    if env is not None:
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                for e in f.iterfind("./Exceptions/Exception"):
                    i1, i2 = int(e.get("p1")), int(e.get("p2"))
                    j1, j2 = _get_idx(env, [i1, i2])
                    j1, j2 = sorted([j1, j2])
                    exception_dict[(j1, j2)]["eps"] = float(e.get("eps"))
                    exception_dict[(j1, j2)]["q"] = float(e.get("q"))
                    exception_dict[(j1, j2)]["sig"] = float(e.get("sig"))

    for (j1, j2), e in exception_dict.items():
        ET.SubElement(
            exceptions,
            "Exception",
            {
                "eps": str(e["eps"]),
                "p1": str(j1),
                "p2": str(j2),
                "q": str(e["q"]),
                "sig": str(e["sig"]),
            },
        )
    return force


def _make_ligand_vdw_forces(lig: ET.Element) -> ET.Element:
    """VDW interactions inside the ligand"""
    formula = [
        "4*eps*((sig/r)^12 - (sig/r)^6)",
    ]
    force = ET.Element(
        "Force",
        {
            "energy": ";".join(formula),
            "name": "CustomBondForce",
            "type": "CustomBondForce",
            "version": "3",
            "usesPeriodic": "0",
        },
    )

    ET.SubElement(force, "GlobalParameters")
    perparticle_parameters = ET.SubElement(force, "PerBondParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "eps"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "sig"}))

    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    particle_params = {}

    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f
            break

    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        eps = float(p.get("eps"))
        sig = float(p.get("sig"))
        particle_params[j] = {
            "eps": eps,
            "sig": sig,
        }

    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f
            break

    bonds = ET.SubElement(force, "Bonds")

    for e in lig_nf.iterfind("./Exceptions/Exception"):
        i1, i2 = int(e.get("p1")), int(e.get("p2"))
        j1, j2 = _get_idx(lig, [i1, i2])
        j1, j2 = sorted([j1, j2])
        eps = float(e.get("eps"))
        sig = float(e.get("sig"))
        ET.SubElement(
            bonds,
            "Bond",
            {
                "p1": str(j1),
                "p2": str(j2),
                "param1": str(eps),
                "param2": str(sig),
            },
        )


    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        for k in range(i + 1, len(lig_nf.findall("./Particles/Particle"))):
            l = int(lig.find("./Particles")[k].get("idx"))
            if (j, l) not in [(int(b.get("p1")), int(b.get("p2"))) for b in bonds.findall("./Bond")]:
                eps = (particle_params[j]["eps"] *
                       particle_params[l]["eps"]) ** 0.5
                sig = 0.5 * (particle_params[j]
                             ["sig"] + particle_params[l]["sig"])
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "p1": str(j),
                        "p2": str(l),
                        "param1": str(eps),
                        "param2": str(sig),
                    },
                )

    return force


def _make_ligand_coul_forces(lig: ET.Element, lambdas: Tuple[float, float]) -> ET.Element:
    """Add back the Coulomb interactions inside the ligand
    """
    formula = [
        "ke * q / r",
    ]
    force = ET.Element(
        "Force",
        {
            "energy": ";".join(formula),
            "name": "CustomBondForce",
            "type": "CustomBondForce",
            "version": "3",
            "usesPeriodic": "0",
        },
    )

    eps_0 = 8.854187817e-12 * unit.farad / unit.meter
    # 138.935485 kJ/mol in nm^2 * mol / e^2
    ke = (1.602176634e-19 ** 2 * 6.0221e23) / (4 * np.pi * eps_0 * 1e-9 * 1000)

    global_parameters = ET.SubElement(force, "GlobalParameters")
    ET.SubElement(global_parameters, "Parameter", {
                  "name": "ke", "default": f'{ke._value}'})
    perparticle_parameters = ET.SubElement(force, "PerBondParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "q"}))

    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    lambda_coul = lambdas[0]


    particle_params = {}


    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f
            break

    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        q = float(p.get("q"))
        particle_params[j] = {
            "charge": q,
        }

    bonds = ET.SubElement(force, "Bonds")


    for e in lig_nf.iterfind("./Exceptions/Exception"):
        i1, i2 = int(e.get("p1")), int(e.get("p2"))
        j1, j2 = _get_idx(lig, [i1, i2])
        j1, j2 = sorted([j1, j2])
        q = float(e.get("q")) * (1 - lambda_coul)
        ET.SubElement(
            bonds,
            "Bond",
            {
                "p1": str(j1),
                "p2": str(j2),
                "param1": str(q),
            },
        )


    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        for k in range(i + 1, len(lig_nf.findall("./Particles/Particle"))):
            l = int(lig.find("./Particles")[k].get("idx"))
            if (j, l) not in [(int(b.get("p1")), int(b.get("p2"))) for b in bonds.findall("./Bond")]:
                q = particle_params[j]["charge"] * \
                    particle_params[l]["charge"] * (1-lambda_coul ** 2)
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "p1": str(j),
                        "p2": str(l),
                        "param1": str(q),
                    },
                )

    return force


def _make_custom_forces(lig: ET.Element, lambdas: Tuple[float, float], env: ET.Element = None) -> ET.Element:
    """Softcore potential between ligand and environment
    """
    formula = [
        "4*epsilon*lambda*(1/(alpha*(1-lambda) + (r/sigma)^6)^2 - 1/(alpha*(1-lambda) + (r/sigma)^6))",
        "epsilon = sqrt(eps1*eps2)",
        "sigma = 0.5*(sig1+sig2)",
        "alpha = 0.5",
    ]
    lig_nf = None
    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            lig_nf = f

    env_nf = None
    if env is not None:
        method = '2'  # 2: CutoffPeriodic
        for f in env.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                env_nf = f

        cut_off = env_nf.get("cutoff")
        switch_distance = env_nf.get("switchingDistance")
        long_range_correction = env_nf.get("dispersionCorrection")
    else:
        method = '1'  # CutOffNonPeriodic
        cut_off = lig_nf.get("cutoff")
        switch_distance = lig_nf.get("switchingDistance")
        long_range_correction = lig_nf.get("dispersionCorrection")

    force = ET.Element(
        "Force",
        {
            "cutoff": cut_off,
            "energy": ";".join(formula),
            "forceGroup": "0",
            "method": method,
            "name": "CustomNonbondedForce",
            "switchingDistance": switch_distance,
            "type": "CustomNonbondedForce",
            "useLongRangeCorrection": long_range_correction,
            "useSwitchingFunction": "1",
            "version": "3",
        },
    )

    perparticle_parameters = ET.SubElement(force, "PerParticleParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "eps"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "sig"}))

    global_parameters = ET.SubElement(force, "GlobalParameters")
    lambda_vdw = lambdas[1]
    ET.SubElement(global_parameters, "Parameter", {
                  "name": "lambda", "default": str(lambda_vdw)})

    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    particles = ET.SubElement(force, "Particles")

    for p in lig.iterfind("./Particles/Particle"):
        ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})

    if env is not None:
        for p in env.iterfind("./Particles/Particle"):
            ET.SubElement(particles, "Particle", {
                          "param1": "0", "param2": "0"})

    for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
        j = int(lig.find("./Particles")[i].get("idx"))
        eps = float(p.get("eps"))
        sig = float(p.get("sig"))
        particles[j].set("param1", str(eps))
        particles[j].set("param2", str(sig))

    if env_nf is not None:
        for i, p in enumerate(env_nf.iterfind("./Particles/Particle")):
            j = int(env.find("./Particles")[i].get("idx"))
            particles[j].set("param1", p.get("eps"))
            particles[j].set("param2", p.get("sig"))

    interaction_groups = ET.SubElement(force, "InteractionGroups")
    group = ET.SubElement(interaction_groups, "InteractionGroup")
    set1 = ET.SubElement(group, "Set1")
    set2 = ET.SubElement(group, "Set2")

    for i, p in enumerate(lig.iterfind("./Particles/Particle")):
        j = p.get("idx")
        ET.SubElement(set1, "Particle", {"index": j})

    if env is not None:
        for i, p in enumerate(env.iterfind("./Particles/Particle")):
            j = p.get("idx")
            ET.SubElement(set2, "Particle", {"index": j})

    exclusions = ET.SubElement(force, "Exclusions")
    exclusion_set = set()

    for f in lig.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            for e in f.iterfind("./Exceptions/Exception"):
                i1, i2 = int(e.get("p1")), int(e.get("p2"))
                j1, j2 = sorted(_get_idx(lig, [i1, i2]))
                if (j1, j2) not in exclusion_set:
                    ET.SubElement(exclusions, "Exclusion", {
                                  "p1": str(j1), "p2": str(j2)})
                    exclusion_set.add((j1, j2))

    if env is not None:
        for e in env_nf.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            j1, j2 = sorted(_get_idx(env, [i1, i2]))
            if (j1, j2) not in exclusion_set:
                ET.SubElement(exclusions, "Exclusion", {
                              "p1": str(j1), "p2": str(j2)})
                exclusion_set.add((j1, j2))

    return force


def _make_cmmotion_remover():
    force = ET.Element(
        "Force",
        {
            "forceGroup": "0",
            "frequency": "1",
            "name": "CMMotionRemover",
            "type": "CMMotionRemover",
            "version": "1",
        },
    )
    return force