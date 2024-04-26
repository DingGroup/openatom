import networkx as nx
import openmm
import xml.etree.ElementTree as ET
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from collections import defaultdict
import re


def make_scaling_factors(n_elec: int, n_vdw: int):
    scaling_factors = []
    for i in range(n_elec + 1):
        scaling_factors.append([(1 - i / n_elec, 1.0), (0.0, 0.0)])

    for i in range(1, n_vdw + 1):
        scaling_factors.append([(0.0, 1 - i / n_vdw), (0.0, i / n_vdw)])

    for i in range(1, n_elec + 1):
        scaling_factors.append([(0.0, 0.0), (i / n_elec, 1.0)])

    return scaling_factors


def make_alchemical_system(
    ligs: list[ET.Element],
    lig_tops: list[openmm.app.Topology],
    lig_common_particles: list[list[int]],
    lig_coors: list[np.ndarray],
    lambdas: list[(float, float)],
    environment: ET.Element,
    environment_top: openmm.app.Topology,
    environment_coor: np.ndarray,
) -> tuple[ET.Element, np.ndarray]:
    """Make an alchemical system from ligands and environment."""

    lig_graphs = [make_graph(lig_top) for lig_top in lig_tops]
    for ligand, common_atoms, graph in zip(ligs, lig_common_particles, lig_graphs):
        label_particles(ligand, common_atoms, graph)

    system = ET.Element("System", environment.attrib)
    system.append(environment.find("./PeriodicBoxVectors"))

    particles, topology = merge_particles_and_topologies(
        ligs, lig_tops, lig_common_particles, environment, environment_top
    )
    system.append(particles)

    constraints = merge_constraints(ligs, environment)
    system.append(constraints)

    forces = ET.SubElement(system, "Forces")

    force = merge_harmonic_bonds(ligs, lambdas, environment)
    forces.append(force)

    force = merge_harmonic_angles(ligs, lambdas, environment)
    forces.append(force)

    force = merge_periodic_torsions(ligs, lambdas, environment)
    forces.append(force)

    force = merge_nonbonded_forces(ligs, lambdas, environment)
    forces.append(force)

    force = make_custom_nonbonded_force(ligs, lambdas, environment)
    forces.append(force)

    force = make_cmmotion_remover()
    forces.append(force)

    for i in range(1, len(lig_coors)):
        lig_coors[i] = align_coordinates(
            lig_coors[0], lig_coors[i], lig_common_particles[0], lig_common_particles[i]
        )
        lig_coors[i][lig_common_particles[i]] = lig_coors[0][lig_common_particles[0]]

    n = len(system.findall("./Particles/Particle"))
    coor = np.zeros((n, 3))
    for idx_lig, (lig, lig_coor) in enumerate(zip(ligs, lig_coors)):
        for i, p in enumerate(lig.iterfind("./Particles/Particle")):
            if idx_lig == 0:
                j = int(p.get("idx"))
                coor[j] = lig_coor[i]
            else:
                if p.get("class") == "soft-core":
                    j = int(p.get("idx"))
                    coor[j] = lig_coor[i]
    for i, p in enumerate(environment.iterfind("./Particles/Particle")):
        j = int(p.get("idx"))
        coor[j] = environment_coor[i]

    coor = np.array(coor)

    return system, topology, coor


def align_coordinates(
    x1: np.ndarray, x2: np.ndarray, atoms1: list[int], atoms2: list[int]
) -> np.ndarray:
    """Align the coordinates of x2 to x1 using the atoms specified in atoms1 and atoms2.

    The coordinates of x2 are first rotated and translated to minimize the RMSD between x1[atom1] and x2[atom2]. Then the coordinates of x2[atom2] are set to x1[atom1].

    Args:
        x1 (np.ndarray): Coordinates of the first molecule.
        x2 (np.ndarray): Coordinates of the second molecule.
        atoms1 (list[int]): List of atom indices of the first molecule.
        atoms2 (list[int]): List of atom indices of the second molecule.

    Returns:
        np.ndarray: Aligned coordinates of the second molecule.

    """
    x1_center = x1[atoms1].mean(axis=0)
    x2_center = x2[atoms2].mean(axis=0)
    x2 = x2 - x2_center
    r = Rotation.align_vectors(x1[atoms1] - x1_center, x2[atoms2])[0]
    return r.apply(x2) + x1_center


def make_graph(topology: openmm.app.Topology) -> nx.Graph:
    """Convert an OpenMM topology to a NetworkX graph.

    The nodes of the graph are atoms and the edges are bonds. Each node has an attribute "element"
    which is the element symbol of the atom. If the atom is a hydrogen, the element attribute is
    the concatenation of the element symbols of the atom that the hydrogen is bonded to and the hydrogen. Each edge has an attribute "type" which is the bond type.

    Args:
        topology (openmm.app.Topology): OpenMM topology object.

    Returns:
        nx.Graph: NetworkX graph of the topology.

    """
    g = nx.Graph()
    for bond in topology.bonds():
        atom1, atom2 = bond.atom1, bond.atom2
        symbol1, symbol2 = atom1.element.symbol, atom2.element.symbol

        if symbol1 == "H":
            g.add_node(atom1.index, element=symbol2 + symbol1)
        else:
            g.add_node(atom1.index, element=symbol1)

        if symbol2 == "H":
            g.add_node(atom2.index, element=symbol1 + symbol2)
        else:
            g.add_node(atom2.index, element=symbol2)

        g.add_edge(atom1.index, atom2.index, type=bond.type)

    return g


def compute_mcs(top1: openmm.app.Topology, top2: openmm.app.Topology) -> dict:
    """Compute the maximum common substructure between two topologies.

    Each topology is converted to a NetworkX graph using the make_graph function and
    the maximum common substructure is computed using the ISMAGS algorithm implemented in NetworkX
    based on the NetworkX graphs of the topologies.

    Args:
        top1 (openmm.app.Topology): OpenMM topology object of the first molecule.
        top2 (openmm.app.Topology): OpenMM topology object of the second molecule.

    Returns:
        dict: A dictionary where the keys are the atom indices of the first molecule and
        the values are the atom indices of the second molecule that are in the common
        substructure.
    """

    if top1.getNumAtoms() >= top2.getNumAtoms():
        top_large, top_small = top1, top2
        key = "first"
    else:
        top_large, top_small = top2, top1
        key = "second"

    graph_large = make_graph(top_large)
    graph_small = make_graph(top_small)
    nm = nx.algorithms.isomorphism.categorical_node_match("element", "")
    em = nx.algorithms.isomorphism.categorical_edge_match("type", "")

    isomag = nx.algorithms.isomorphism.ISMAGS(
        graph_large, graph_small, node_match=nm, edge_match=em
    )
    lcss = list(isomag.largest_common_subgraph())

    if len(lcss) > 1:
        Warning(
            "More than one largest common substructures found. Returning the first one."
        )

    lcs = lcss[0]
    subgraph_large = graph_large.subgraph(list(lcs.keys()))

    components = list(nx.connected_components(subgraph_large))
    len_components = [len(c) for c in components]
    largest_component = components[np.argmax(len_components)]

    connected_lcs = {i: lcs[i] for i in largest_component}

    if key == "first":
        return connected_lcs
    else:
        return {v: k for k, v in connected_lcs.items()}


def label_particles(
    ligand: ET.Element, common_particles: list[int], graph: nx.Graph
) -> None:
    """Label the particles in a ligand based on the common substructure and the graph of the ligand.

    Every particle has a "class" attribute which can be either "common" or "soft-core". The common
    particles are the particles that are in the common substructure. The soft-core particles are
    the particles that are not in the common substructure. For each soft-core particle, it has an
    "attach_idx" attribute which is the index of the common particle through which it is connected
    to the common substructure.

    Args:
        ligand (ET.Element): XML element of the ligand.
            It should be the root of the ligand XML tree.
        common_particles (list[int]): List of indices of common particles in the ligand.
        graph (nx.Graph): NetworkX graph of the ligand.

    Note that the function modifies the input ligand and does not return anything.

    """

    ## label particles' class as common or soft-core
    for i, p in enumerate(ligand.iterfind("./Particles/Particle")):
        if i in common_particles:
            p.set("class", "common")
        else:
            p.set("class", "soft-core")

    ## label soft-core particles' attach_idx using breadth-first search
    for i, j in nx.bfs_edges(graph, source=common_particles[0]):
        if (i in common_particles) and (j not in common_particles):
            ligand.find("./Particles")[j].set("attach_idx", str(i))
        elif (i not in common_particles) and (j not in common_particles):
            ligand.find("./Particles")[j].set(
                "attach_idx", ligand.find("./Particles")[i].get("attach_idx")
            )


def merge_particles_and_topologies(
    ligands: list[ET.Element],
    ligand_topologies: list[openmm.app.Topology],
    common_particles: list[list[int]],
    environment: ET.Element,
    environment_topology: openmm.app.Topology,
) -> ET.Element:
    """Merge the particles and topologies of ligands and environment.

    Because ligands share common particles, the common particles are only added once to the
    merged particles. Soft-core particles are added from all ligands. The environment particles
    are added as well. As the particles are added, their indices in the merged particles are
    recorded in the "idx" attribute of the particles in the ligands and the environment, which
    is the intended side effect of this function.

    Args:
        ligands (list[ET.Element]): List of XML elements of the ligands.
            Each element should be the root of the ligand XML tree.
        ligand_topologies (list[openmm.app.Topology]): List of OpenMM topology objects of the
            ligands.
        common_particles (list[list[int]]): List of lists of indices of common particles in the
            ligands.
        environment (ET.Element): XML element of the environment.
            It should be the root of the environment XML tree.
        environment_topology (openmm.app.Topology): OpenMM topology object of the environment.

    Returns:
        ET.Element: XML element of the merged particles.
        openmm.app.Topology: OpenMM topology object of the merged system.

    Note that the function has an intended side effect of setting the "idx" attribute of the
    particles in the ligands and the environment.

    """

    particles = ET.Element("Particles")
    topology = openmm.app.Topology()

    ligand_atoms = [list(lig_top.atoms()) for lig_top in ligand_topologies]

    ## add common particles from the first ligand and label common particles in all ligands
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)

    p_idx = 0
    for i in range(len(common_particles[0])):
        lig = ligands[0]
        j = common_particles[0][i]
        ET.SubElement(
            particles, "Particle", {"mass": lig.find("./Particles")[j].get("mass")}
        )
        lig.find("./Particles")[j].set("idx", str(p_idx))

        atom = ligand_atoms[0][j]
        topology.addAtom(atom.name, atom.element, residue)

        for k in range(1, len(ligands)):
            lig = ligands[k]
            j = common_particles[k][i]
            lig.find("./Particles")[j].set("idx", str(p_idx))

        p_idx += 1

    ## add soft-core particles from all ligands
    atom_names = set()
    for lig, atoms in zip(ligands, ligand_atoms):
        for p, atom in zip(lig.iterfind("./Particles/Particle"), atoms):
            if p.get("class") == "soft-core":
                ET.SubElement(particles, "Particle", {"mass": p.get("mass")})
                p.set("idx", str(p_idx))
                p_idx += 1

                ## soft-core particles from different ligands may have the same name
                ## so we need to check and increment the name if necessary to avoid
                ## name conflicts in the topology. This is necessary only for topology
                if atom.name not in atom_names:
                    topology.addAtom(atom.name, atom.element, residue)
                    atom_names.add(atom.name)
                else:
                    name = atom.name
                    while name in atom_names:
                        name = increment_atom_name(name)
                    topology.addAtom(name, atom.element, residue)
                    atom_names.add(name)

    ## add envirionment particles
    chain = topology.addChain()
    residue = topology.addResidue("ENV", chain)

    for chain in environment_topology.chains():
        new_chain = topology.addChain(id=chain.id)
        for residue in chain.residues():
            new_residue = topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                i = int(atom.index)
                p = environment.find("./Particles")[i]
                particles.append(copy.deepcopy(p))
                p.set("idx", str(p_idx))
                p_idx += 1

                topology.addAtom(atom.name, atom.element, new_residue)

    ## add bonds to the topology
    topology_atoms = list(topology.atoms())
    for lig, lig_top in zip(ligands, ligand_topologies):
        for bond in lig_top.bonds():
            i, j = bond.atom1.index, bond.atom2.index
            i = int(lig.find("./Particles")[i].get("idx"))
            j = int(lig.find("./Particles")[j].get("idx"))
            topology.addBond(topology_atoms[i], topology_atoms[j])

    for bond in environment_topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        i = int(environment.find("./Particles")[i].get("idx"))
        j = int(environment.find("./Particles")[j].get("idx"))
        topology.addBond(topology_atoms[i], topology_atoms[j])

    ## set periodic box vectors
    topology.setPeriodicBoxVectors(environment_topology.getPeriodicBoxVectors())

    return particles, topology


def merge_constraints(ligands: list[ET.Element], environment: ET.Element) -> ET.Element:
    """Merge the constraints of ligands and environment.

    The constraints of the ligands among common particles are added once to the merged constraints
    using the constraints from the first ligand. The constraints involving soft-core particles
    are added from all ligands. The constraints from the environment are added as well.

    Args:
        ligands (list[ET.Element]): List of XML elements of the ligands.
            Each element should be the root of the ligand XML tree.
        environment (ET.Element): XML element of the environment.
            It should be the root of the environment XML tree.

    Returns:
        ET.Element: XML element of the merged constraints.

    """

    cons = ET.Element("Constraints")

    ## add constraints from the ligands
    for idx_lig, lig in enumerate(ligands):
        for c in lig.iterfind("./Constraints/Constraint"):
            i1, i2 = int(c.get("p1")), int(c.get("p2"))
            j1, j2 = _get_idx(lig, [i1, i2])
            j1, j2 = str(j1), str(j2)

            ## all all constraints from the first ligand
            if idx_lig == 0:
                ET.SubElement(cons, "Constraint", {"d": c.get("d"), "p1": j1, "p2": j2})

            ## only add constraints involving soft-core particles from other ligands
            else:
                if (
                    lig.find("./Particles")[i1].get("class") == "soft-core"
                    or lig.find("./Particles")[i2].get("class") == "soft-core"
                ):
                    ET.SubElement(
                        cons, "Constraint", {"d": c.get("d"), "p1": j1, "p2": j2}
                    )

    ## add constraints from the environment
    for c in environment.iterfind("./Constraints/Constraint"):
        i1, i2 = int(c.get("p1")), int(c.get("p2"))
        j1, j2 = _get_idx(environment, [i1, i2])
        j1, j2 = str(j1), str(j2)
        ET.SubElement(cons, "Constraint", {"d": c.get("d"), "p1": j1, "p2": j2})

    return cons


def _get_idx(root: ET.Element, idxs: list[int]) -> list[int]:
    return [int(root.find("./Particles")[i].get("idx")) for i in idxs]


def merge_harmonic_bonds(
    ligands: list[ET.Element], lambdas: list[(float, float)], environment: ET.Element
) -> ET.Element:
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

    ligand_bonds = []
    for ligand in ligands:
        for f in ligand.iterfind("./Forces/Force"):
            if f.get("type") == "HarmonicBondForce":
                ligand_bonds.append(f)
    assert len(ligand_bonds) == len(ligands)

    env_bonds = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "HarmonicBondForce":
            env_bonds = f

    for (_, lambda_vdw), ligand_bond, ligand in zip(lambdas, ligand_bonds, ligands):
        for b in ligand_bond.iterfind("./Bonds/Bond"):
            i1, i2 = int(b.get("p1")), int(b.get("p2"))
            j1, j2 = _get_idx(ligand, [i1, i2])
            j1, j2 = str(j1), str(j2)

            if (
                ligand.find("./Particles")[i1].get("class") == "common"
                and ligand.find("./Particles")[i2].get("class") == "common"
            ):
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "d": b.get("d"),
                        "k": str(float(b.get("k")) * lambda_vdw),
                        "p1": j1,
                        "p2": j2,
                    },
                )
            else:
                ET.SubElement(
                    bonds,
                    "Bond",
                    {"d": b.get("d"), "k": str(float(b.get("k"))), "p1": j1, "p2": j2},
                )

    for b in env_bonds.iterfind("./Bonds/Bond"):
        i1, i2 = int(b.get("p1")), int(b.get("p2"))
        j1, j2 = _get_idx(environment, [i1, i2])
        j1, j2 = str(j1), str(j2)
        ET.SubElement(
            bonds, "Bond", {"d": b.get("d"), "k": b.get("k"), "p1": j1, "p2": j2}
        )

    return force


def merge_harmonic_angles(
    ligands: list[ET.Element],
    lambdas: list[(float, float)],
    environment: ET.Element,
) -> ET.Element:
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

    ligand_angles = []
    for ligand in ligands:
        for f in ligand.iterfind("./Forces/Force"):
            if f.get("type") == "HarmonicAngleForce":
                ligand_angles.append(f)
    assert len(ligand_angles) == len(ligands)

    env_angles = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "HarmonicAngleForce":
            env_angles = f

    for (_, lambda_vdw), ligand_angle, ligand in zip(lambdas, ligand_angles, ligands):
        for a in ligand_angle.iterfind("./Angles/Angle"):
            i1, i2, i3 = int(a.get("p1")), int(a.get("p2")), int(a.get("p3"))
            j1, j2, j3 = _get_idx(ligand, [i1, i2, i3])

            if (
                ligand.find("./Particles")[i1].get("class") == "common"
                and ligand.find("./Particles")[i2].get("class") == "common"
                and ligand.find("./Particles")[i3].get("class") == "common"
            ):
                ET.SubElement(
                    angles,
                    "Angle",
                    {
                        "a": str(float(a.get("a"))),
                        "k": str(float(a.get("k")) * lambda_vdw),
                        "p1": str(j1),
                        "p2": str(j2),
                        "p3": str(j3),
                    },
                )
            else:
                ET.SubElement(
                    angles,
                    "Angle",
                    {
                        "a": str(float(a.get("a"))),
                        "k": str(float(a.get("k"))),
                        "p1": str(j1),
                        "p2": str(j2),
                        "p3": str(j3),
                    },
                )

    for a in env_angles.iterfind("./Angles/Angle"):
        i1, i2, i3 = int(a.get("p1")), int(a.get("p2")), int(a.get("p3"))
        j1, j2, j3 = _get_idx(environment, [i1, i2, i3])

        ET.SubElement(
            angles,
            "Angle",
            {
                "a": str(float(a.get("a"))),
                "k": str(float(a.get("k"))),
                "p1": str(j1),
                "p2": str(j2),
                "p3": str(j3),
            },
        )

    return force


def merge_periodic_torsions(
    ligands: list[ET.Element],
    lambdas: list[(float, float)],
    environment: ET.Element,
) -> ET.Element:
    force = ET.Element(
        "Force",
        {
            "forceGroup": "0",
            "name": "PeridociTorsionForce",
            "type": "PeriodicTorsionForce",
            "usesPeriodic": "0",
            "version": "2",
        },
    )
    torsions = ET.SubElement(force, "Torsions")

    ligand_torsions = []
    for ligand in ligands:
        for f in ligand.iterfind("./Forces/Force"):
            if f.get("type") == "PeriodicTorsionForce":
                ligand_torsions.append(f)
    assert len(ligand_torsions) == len(ligands)

    env_torsion = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "PeriodicTorsionForce":
            env_torsion = f

    for (_, lambda_vdw), ligand_torsion, ligand in zip(lambdas, ligand_torsions, ligands):
        for t in ligand_torsion.iterfind("./Torsions/Torsion"):
            i1, i2, i3, i4 = map(
                int, [t.get("p1"), t.get("p2"), t.get("p3"), t.get("p4")]
            )
            j1, j2, j3, j4 = _get_idx(ligand, [i1, i2, i3, i4])

            if (
                ligand.find("./Particles")[i1].get("class") == "common"
                and ligand.find("./Particles")[i2].get("class") == "common"
                and ligand.find("./Particles")[i3].get("class") == "common"
                and ligand.find("./Particles")[i4].get("class") == "common"
            ):
                ET.SubElement(
                    torsions,
                    "Torsion",
                    {
                        "k": str(float(t.get("k")) * lambda_vdw),
                        "p1": str(j1),
                        "p2": str(j2),
                        "p3": str(j3),
                        "p4": str(j4),
                        "periodicity": t.get("periodicity"),
                        "phase": t.get("phase"),
                    },
                )
            else:
                ET.SubElement(
                    torsions,
                    "Torsion",
                    {
                        "k": str(float(t.get("k"))),
                        "p1": str(j1),
                        "p2": str(j2),
                        "p3": str(j3),
                        "p4": str(j4),
                        "periodicity": t.get("periodicity"),
                        "phase": t.get("phase"),
                    },
                )

    for t in env_torsion.iterfind("./Torsions/Torsion"):
        i1, i2, i3, i4 = map(int, [t.get("p1"), t.get("p2"), t.get("p3"), t.get("p4")])
        j1, j2, j3, j4 = _get_idx(environment, [i1, i2, i3, i4])

        ET.SubElement(
            torsions,
            "Torsion",
            {
                "k": str(float(t.get("k"))),
                "p1": str(j1),
                "p2": str(j2),
                "p3": str(j3),
                "p4": str(j4),
                "periodicity": t.get("periodicity"),
                "phase": t.get("phase"),
            },
        )

    return force


def merge_nonbonded_forces(
    ligands: list[ET.Element],
    lambdas: list,
    environment: ET.Element,
) -> ET.Element:
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            attrib = f.attrib
    attrib["dispersionCorrection"] = "0"

    force = ET.Element("Force", attrib)
    ET.SubElement(force, "GlobalParameters")
    ET.SubElement(force, "ParticleOffsets")
    ET.SubElement(force, "ExceptionOffsets")

    particles = ET.SubElement(force, "Particles")

    for i, ligand in enumerate(ligands):
        for p in ligand.iterfind("./Particles/Particle"):
            if i == 0:
                ET.SubElement(
                    particles, "Particle", {"eps": "0.0", "q": "0.0", "sig": "0.0"}
                )
            if i > 0 and p.get("class") == "soft-core":
                ET.SubElement(
                    particles,
                    "Particle",
                    {"eps": "0.0", "q": "0.0", "sig": "0.0"},
                )

    for p in environment.iterfind("./Particles/Particle"):
        ET.SubElement(particles, "Particle", {"eps": "0.0", "q": "0.0", "sig": "0.0"})

    ligand_nfs = []
    for ligand in ligands:
        for f in ligand.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                ligand_nfs.append(f)
    assert len(ligand_nfs) == len(ligands)

    env_nf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            env_nf = f

    for (lambda_elec, lambda_vdw), lig_nf, ligand in zip(lambdas, ligand_nfs, ligands):
        for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
            j = int(ligand.find("./Particles")[i].get("idx"))

            ## if the ligand particle is a common particle, scale eps, sig and q using svdw
            if ligand.find("./Particles")[i].get("class") == "common":
                eps = float(p.get("eps")) * lambda_vdw
                q = float(p.get("q")) * lambda_vdw
                sigma = float(p.get("sig")) * lambda_vdw

                current_eps = float(particles[j].get("eps"))
                current_sigma = float(particles[j].get("sig"))
                current_q = float(particles[j].get("q"))

                particles[j].set("eps", str(current_eps + eps))
                particles[j].set("q", str(current_q + q))
                particles[j].set("sig", str(current_sigma + sigma))

            ## if the ligand particle is a soft-core particle, set eps and sig to 0 and scale its
            ## charge using selec. Also move the charge to the common particle through which it
            ## is connected to the common substructure.
            else:
                particles[j].set("eps", "0")
                particles[j].set("sig", "1")

                q = float(p.get("q"))

                ## set the charge of the alchemical particle
                particles[j].set("q", str(q * lambda_elec))

                ## set the charge of the attached common particle
                k = ligand.find("./Particles")[i].get("attach_idx")
                j = int(ligand.find("./Particles")[int(k)].get("idx"))
                current_q = float(particles[j].get("q"))
                particles[j].set("q", str(current_q + (1 - lambda_elec) * q * lambda_vdw))

    for i, p in enumerate(env_nf.iterfind("./Particles/Particle")):
        j = int(environment.find("./Particles")[i].get("idx"))
        particles[j].set("eps", p.get("eps"))
        particles[j].set("sig", p.get("sig"))
        particles[j].set("q", p.get("q"))

    ## exceptions
    exception_dict = defaultdict(lambda: {"eps": 0, "q": 0, "sig": 0})
    for (lambda_elec, lambda_vdw), lig_nf, ligand in zip(lambdas, ligand_nfs, ligands):
        for e in lig_nf.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            j1, j2 = _get_idx(ligand, [i1, i2])
            j1, j2 = sorted([j1, j2])

            if (
                ligand.find("./Particles")[i1].get("class") == "common"
                and ligand.find("./Particles")[i2].get("class") == "common"
            ):
                exception_dict[(j1, j2)]["eps"] += float(e.get("eps")) * lambda_vdw
                exception_dict[(j1, j2)]["q"] += float(e.get("q")) * lambda_vdw
                exception_dict[(j1, j2)]["sig"] += float(e.get("sig")) * lambda_vdw

            elif (
                ligand.find("./Particles")[i1].get("class") == "soft-core"
                and ligand.find("./Particles")[i2].get("class") == "soft-core"
            ):
                exception_dict[(j1, j2)]["eps"] = float(e.get("eps"))
                exception_dict[(j1, j2)]["q"] = float(e.get("q"))
                exception_dict[(j1, j2)]["sig"] = float(e.get("sig"))

            else:
                exception_dict[(j1, j2)]["eps"] = float(e.get("eps")) * lambda_vdw
                exception_dict[(j1, j2)]["q"] = float(e.get("q")) * lambda_vdw
                exception_dict[(j1, j2)]["sig"] = float(e.get("sig"))

    for e in env_nf.iterfind("./Exceptions/Exception"):
        i1, i2 = int(e.get("p1")), int(e.get("p2"))
        j1, j2 = _get_idx(environment, [i1, i2])
        j1, j2 = sorted([j1, j2])
        exception_dict[(j1, j2)]["eps"] = float(e.get("eps"))
        exception_dict[(j1, j2)]["q"] = float(e.get("q"))
        exception_dict[(j1, j2)]["sig"] = float(e.get("sig"))

    exceptions = ET.SubElement(force, "Exceptions")
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


def make_custom_nonbonded_force(
    ligands: list[ET.Element],
    lambdas: list,
    environment: ET.Element,
) -> ET.Element:
    formula = [
        "4*epsilon*lambda*(1/(alpha*(1-lambda) + (r/sigma)^6)^2 - 1/(alpha*(1-lambda) + (r/sigma)^6))",
        "epsilon = sqrt(eps1*eps2)",
        "sigma = 0.5*(sig1+sig2)",
        "alpha = 0.5",
        "lambda = select(group1*group2, lambda_b ,lambda_a)",
        "lambda_b = select(group1 - group2, 0, 1)",
        "lambda_a = lambda1 + lambda2",
    ]

    force = ET.Element(
        "Force",
        {
            "cutoff": "1.2",
            "energy": ";".join(formula),
            "forceGroup": "0",
            "method": "2",
            "name": "CustomNonbondedForce",
            "switchingDistance": "1.0",
            "type": "CustomNonbondedForce",
            "useLongRangeCorrection": "0",
            "useSwitchingFunction": "1",
            "version": "3",
        },
    )

    perparticle_parameters = ET.SubElement(force, "PerParticleParameters")
    perparticle_parameters.append(ET.Element("Parameter", {"name": "eps"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "sig"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "lambda"}))
    perparticle_parameters.append(ET.Element("Parameter", {"name": "group"}))

    ET.SubElement(force, "GlobalParameters")
    ET.SubElement(force, "ComputedValues")
    ET.SubElement(force, "EnergyParameterDerivatives")
    ET.SubElement(force, "Functions")

    particles = ET.SubElement(force, "Particles")

    for i, ligand in enumerate(ligands):
        for p in ligand.iterfind("./Particles/Particle"):
            if i == 0:
                ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})
            if i > 0 and p.get("class") == "soft-core":
                ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})
    for p in environment.iterfind("./Particles/Particle"):
        ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})

    lig_nfs = []
    for ligand in ligands:
        for f in ligand.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                lig_nfs.append(f)
    assert len(lig_nfs) == len(ligands)

    env_nf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            env_nf = f

    for idx_lig, ((_, lambda_vdw), lig_nf, ligand) in enumerate(
        zip(lambdas, lig_nfs, ligands)
    ):
        for i, p in enumerate(lig_nf.iterfind("./Particles/Particle")):
            j = int(ligand.find("./Particles")[i].get("idx"))
            eps = float(p.get("eps"))
            sigma = float(p.get("sig"))
            if ligand.find("./Particles")[i].get("class") == "common":
                current_eps = float(particles[j].get("param1"))
                current_sigma = float(particles[j].get("param2"))

                particles[j].set("param1", str(current_eps + eps * lambda_vdw))
                particles[j].set("param2", str(current_sigma + sigma * lambda_vdw))
                particles[j].set("param3", "0")
                particles[j].set("param4", "0")

            else:
                particles[j].set("param1", str(eps))
                particles[j].set("param2", str(sigma))
                particles[j].set("param3", str(lambda_vdw))
                particles[j].set("param4", str(idx_lig + 1))

    for i, p in enumerate(env_nf.iterfind("./Particles/Particle")):
        j = int(environment.find("./Particles")[i].get("idx"))
        particles[j].set("param1", p.get("eps"))
        particles[j].set("param2", p.get("sig"))
        particles[j].set("param3", "0")
        particles[j].set("param4", "0")

    interaction_groups = ET.SubElement(force, "InteractionGroups")
    interaction_group = ET.SubElement(interaction_groups, "InteractionGroup")
    set1 = ET.SubElement(interaction_group, "Set1")
    set2 = ET.SubElement(interaction_group, "Set2")
    for i, ligand in enumerate(ligands):
        for p in ligand.iterfind("./Particles/Particle"):
            j = p.get("idx")
            if i == 0 and p.get("class") == "common":
                ET.SubElement(set2, "Particle", {"index": j})
            if i == 0 and p.get("class") == "soft-core":
                ET.SubElement(set1, "Particle", {"index": j})
            if i > 0 and p.get("class") == "soft-core":
                ET.SubElement(set1, "Particle", {"index": j})

    for p in environment.iterfind("./Particles/Particle"):
        j = p.get("idx")
        ET.SubElement(set2, "Particle", {"index": j})

    exclusions = ET.SubElement(force, "Exclusions")
    lig_nfs = []
    for ligand in ligands:
        for f in ligand.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                lig_nfs.append(f)
    assert len(lig_nfs) == len(ligands)

    for lig_nf, ligand in zip(lig_nfs, ligands):
        for e in lig_nf.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            if (
                ligand.find("./Particles")[i1].get("class") == "common"
                and ligand.find("./Particles")[i2].get("class") == "common"
            ):
                continue
            j1, j2 = _get_idx(ligand, [i1, i2])
            ET.SubElement(exclusions, "Exclusion", {"p1": str(j1), "p2": str(j2)})

    return force


def make_cmmotion_remover():
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


def increment_atom_name(atom_name):
    # Use regular expressions to find all non-digit and digit parts of the string
    match = re.match(r"([a-zA-Z]+)(\d+)", atom_name)
    if match:
        letters = match.group(1)
        digits = match.group(2)
        # Increment the number part
        new_number = str(int(digits) + 1).zfill(
            len(digits)
        )  # zfill keeps the leading zeros
        return letters + new_number
    else:
        # Return original if no digits found
        return atom_name
