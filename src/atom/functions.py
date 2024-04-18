import mdtraj
import networkx as nx
import openmm as omm
import openmm.unit as unit
import xml.etree.ElementTree as ET
import copy


def label_particles(ligand: ET.Element, common_particles: list[int], graph: nx.Graph):
    """Two kinds of labels are added to particles in the ligand:

    1. For each particle in the ligand, create a new attribute "class" and
    set it to "common" if the particle is in the common substructure, otherwise set it to "soft-core".

    2. For each particle with class "soft-core", create a new attribute "attach_idx" and
    set it to the index of the common particle through which it is connected to the common
    substructure.

    Args:
        ligand (ET.Element): XML element of the ligand. It should be the root of the
            ligand XML tree.
        common_particles (list[int]]): List of indices of common particles
        graph (nx.Graph): NetworkX graph of the ligand

    """

    for i, p in enumerate(ligand.iterfind("./Particles/Particle")):
        if i in common_particles:
            p.set("class", "common")
        else:
            p.set("class", "soft-core")

    for i, j in nx.bfs_edges(graph, source=common_particles[0]):
        if (i in common_particles) and (j not in common_particles):
            ligand.find("./Particles")[j].set("attach_idx", str(i))
        elif (i not in common_particles) and (j not in common_particles):
            ligand.find("./Particles")[j].set(
                "attach_idx", ligand.find("./Particles")[i].get("attach_idx")
            )


def merge_and_index_particles(
    system: ET.Element,
    environment: ET.Element,
    ligands: list[ET.Element],
    common_particles: list[list[int]],
):
    """Merge particles from the environment and ligands into the target system.

    The common particles are added from the first ligand.
    Soft-core particles are added from all ligands.
    All particles from the environment are added.

    As particles from ligands and environment are added, their indices in the target system
    are recorded in the "idx" attribute of the corresponding particle. This is useful for
    later merging of forces.

    Args:
        system (ET.Element): XML element of the system.
            It should be the root of the system XML tree.
        environment (ET.Element): XML element of the environment.
            It should be the root of the environment XML tree.
        ligands (list[ET.Element]): List of XML elements of ligands.
            Each element should be the root of the ligand XML tree.
        common_particles (list[list[int]]): List of list of indices of common particles in
            each ligand. The order of the list should be the same as the order of ligands.
            Moverover, the order of indices in each list should match each other, meaning that
            the ith index in each list should correspond to the same common particle.
    """

    particles = ET.SubElement(system, "Particles")

    ## add common particles from the first ligand and label common particles in all ligands
    p_idx = 0
    for i in range(len(common_particles[0])):
        lig = ligands[0]
        j = common_particles[0][i]
        ET.SubElement(
            particles,
            "Particle",
            {"mass": lig.find("./Particles")[j].get("mass")},
        )
        lig.find("./Particles")[j].set("idx", str(p_idx))

        for k in range(1, len(ligands)):
            lig = ligands[k]
            j = common_particles[k][i]
            lig.find("./Particles")[j].set("idx", str(p_idx))

        p_idx += 1

    ## add alchemical atoms from all ligands
    for lig in ligands:
        for p in lig.iterfind("./Particles/Particle"):
            if p.get("class") == "soft-core":
                ET.SubElement(
                    particles,
                    "Particle",
                    {"mass": p.get("mass")},
                )
                p.set("idx", str(p_idx))
                p_idx += 1

    ## add envirionment particles
    for p in environment.iterfind("./Particles/Particle"):
        particles.append(copy.deepcopy(p))
        p.set("idx", str(p_idx))
        p_idx += 1


def merge_constraints(
    system: ET.Element, environment: ET.Element, ligands: list[ET.Element]
):
    constraints = ET.SubElement(system, "Constraints")

    for root in ligands + [environment]:
        for c in root.iterfind("./Constraints/Constraint"):
            cc = copy.deepcopy(c)
            for n in range(1, 5):
                if f"p{n}" in cc.attrib:
                    current_idx = int(cc.get(f"p{n}"))
                    new_idx = root.find("./Particles")[current_idx].get("idx")
                    cc.set(f"p{n}", new_idx)
            constraints.append(cc)


def _get_idx(root: ET.Element, idxs: list[int]) -> list[int]:
    return [int(root.find("./Particles")[i].get("idx")) for i in idxs]
    

def merge_harmonic_bonds(
    system: ET.Element,
    environment: ET.Element,
    ligands: list[ET.Element],
    scaling_factors: list[(float, float)],
):
    if system.find("./Forces") is None:
        forces = ET.SubElement(system, "Forces")
    else:
        forces = system.find("./Forces")

    force = ET.SubElement(
        forces,
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

    lig_hbfs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "HarmonicBondForce":
                lig_hbfs.append(f)
    assert len(lig_hbfs) == len(ligands)

    env_hbf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "HarmonicBondForce":
            env_hbf = f

    for (selec, svdw), lig_hbf, lig in zip(scaling_factors, lig_hbfs, ligands):
        for b in lig_hbf.iterfind("./Bonds/Bond"):
            i1, i2 = int(b.get("p1")), int(b.get("p2"))
            j1, j2 = _get_idx(lig, [i1, i2])

            if (
                lig.find("./Particles")[i1].get("class") == "common"
                and lig.find("./Particles")[i2].get("class") == "common"
            ):
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "d": str(float(b.get("d"))),
                        "k": str(float(b.get("k")) * svdw),
                        "p1": str(j1),
                        "p2": str(j2),
                    },
                )
            else:
                ET.SubElement(
                    bonds,
                    "Bond",
                    {
                        "d": str(float(b.get("d"))),
                        "k": str(float(b.get("k"))),
                        "p1": str(j1),
                        "p2": str(j2),
                    },
                )

    for b in env_hbf.iterfind("./Bonds/Bond"):
        i1, i2 = int(b.get("p1")), int(b.get("p2"))
        j1, j2 = _get_idx(environment, [i1, i2])

        ET.SubElement(
            bonds,
            "Bond",
            {
                "d": str(float(b.get("d"))),
                "k": str(float(b.get("k"))),
                "p1": str(j1),
                "p2": str(j2),
            },
        )

def merge_harmonic_angles(
    system: ET.Element, environment: ET.Element, ligands: list[ET.Element], scaling_factors: list[(float, float)]
):
    if system.find("./Forces") is None:
        forces = ET.SubElement(system, "Forces")
    else:
        forces = system.find("./Forces")

    force = ET.SubElement(
        forces,
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

    lig_hafs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "HarmonicAngleForce":
                lig_hafs.append(f)
    assert len(lig_hafs) == len(ligands)

    env_haf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "HarmonicAngleForce":
            env_haf = f

    for (selec, svdw), lig_haf, lig in zip(scaling_factors, lig_hafs, ligands):
        for a in lig_haf.iterfind("./Angles/Angle"):
            i1, i2, i3 = int(a.get("p1")), int(a.get("p2")), int(a.get("p3"))
            j1, j2, j3 = _get_idx(lig, [i1, i2, i3])

            if (
                lig.find("./Particles")[i1].get("class") == "common"
                and lig.find("./Particles")[i2].get("class") == "common"
                and lig.find("./Particles")[i3].get("class") == "common"
            ):
                ET.SubElement(
                    angles,
                    "Angle",
                    {
                        "a": str(float(a.get("a"))),
                        "k": str(float(a.get("k")) * svdw),
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

    for a in env_haf.iterfind("./Angles/Angle"):
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


def merge_periodic_torsions(
    target: ET.Element, environment: ET.Element, ligands: list[ET.Element], scaling_factors: list[(float, float)]
):
    if target.find("./Forces") is None:
        forces = ET.SubElement(target, "Forces")
    else:
        forces = target.find("./Forces")

    force = ET.SubElement(
        forces,
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

    lig_ptfs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "PeriodicTorsionForce":
                lig_ptfs.append(f)
    assert len(lig_ptfs) == len(ligands)

    env_ptf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "PeriodicTorsionForce":
            env_ptf = f

    for (selec, svdw), lig_ptf, lig in zip(scaling_factors, lig_ptfs, ligands):
        for t in lig_ptf.iterfind("./Torsions/Torsion"):
            i1, i2, i3, i4 = int(t.get("p1")), int(t.get("p2")), int(t.get("p3")), int(t.get("p4"))
            j1, j2, j3, j4 = _get_idx(lig, [i1, i2, i3, i4])

            if (
                lig.find("./Particles")[i1].get("class") == "common"
                and lig.find("./Particles")[i2].get("class") == "common"
                and lig.find("./Particles")[i3].get("class") == "common"
                and lig.find("./Particles")[i4].get("class") == "common"
            ):
                ET.SubElement(
                    torsions,
                    "Torsion",
                    {
                        "k": str(float(t.get("k")) * svdw),
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

    for t in env_ptf.iterfind("./Torsions/Torsion"):
        i1, i2, i3, i4 = int(t.get("p1")), int(t.get("p2")), int(t.get("p3")), int(t.get("p4"))
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



def merge_nonbonded_forces(
    target: ET.Element,
    environment: ET.Element,
    ligands: list[ET.Element],
    scaling_factors: list,
):
    if target.find("./Forces") is None:
        forces = ET.SubElement(target, "Forces")
    else:
        forces = target.find("./Forces")

    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            attrib = f.attrib

    target_nbf = ET.SubElement(forces, "Force", attrib)
    target_particles = ET.SubElement(target_nbf, "Particles")

    for _ in target.iterfind("./Particles/Particle"):
        ET.SubElement(
            target_particles, "Particle", {"eps": "0.0", "sig": "0.0", "q": "0.0"}
        )

    lig_nbfs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                lig_nbfs.append(f)
    assert len(lig_nbfs) == len(ligands)

    env_nbf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            env_nbf = f

    for (selec, svdw), lig_nbf, lig in zip(scaling_factors, lig_nbfs, ligands):
        for i, p in enumerate(lig_nbf.iterfind("./Particles/Particle")):
            j = int(lig.find("./Particles")[i].get("idx"))
            if lig.find("./Particles")[i].get("class") == "common":
                eps = float(p.get("eps")) * svdw
                sigma = float(p.get("sig")) * svdw
                q = float(p.get("q")) * svdw

                current_eps = float(target_particles[j].get("eps"))
                current_sigma = float(target_particles[j].get("sig"))
                current_q = float(target_particles[j].get("q"))

                target_particles[j].set("eps", str(current_eps + eps))
                target_particles[j].set("sig", str(current_sigma + sigma))
                target_particles[j].set("q", str(current_q + q))

            else:
                target_particles[j].set("eps", "0")
                target_particles[j].set("sig", "1")

                q = float(p.get("q"))

                ## set the charge of the alchemical particle
                target_particles[j].set("q", str(q * selec))

                ## set the charge of the attached common particle
                k = lig.find("./Particles")[i].get("attach_idx")
                j = int(lig.find("./Particles")[int(k)].get("idx"))
                current_q = float(target_particles[j].get("q"))
                target_particles[j].set("q", str(current_q + (1 - selec) * q * svdw))

    for i, p in enumerate(env_nbf.iterfind("./Particles/Particle")):
        j = int(environment.find("./Particles")[i].get("idx"))
        target_particles[j].set("eps", p.get("eps"))
        target_particles[j].set("sig", p.get("sig"))
        target_particles[j].set("q", p.get("q"))

    ## exceptions
    target_exceptions = ET.SubElement(target_nbf, "Exceptions")
    for (selec, svdw), lig_nbf, lig in zip(scaling_factors, lig_nbfs, ligands):
        for e in lig_nbf.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            j1 = int(lig.find("./Particles")[i1].get("idx"))
            j2 = int(lig.find("./Particles")[i2].get("idx"))


def get_graph(topology: mdtraj.Topology) -> nx.Graph:
    g = nx.Graph()
    for bond in topology.bonds:
        atom1, atom2 = bond.atom1, bond.atom2
        if atom1.element.symbol == "H":
            g.add_node(atom1.index, element=atom2.element.symbol + atom1.element.symbol)
        else:
            g.add_node(atom1.index, element=atom1.element.symbol)

        if atom2.element.symbol == "H":
            g.add_node(atom2.index, element=atom1.element.symbol + atom2.element.symbol)
        else:
            g.add_node(atom2.index, element=atom2.element.symbol)

        g.add_edge(atom1.index, atom2.index, type=bond.type)

    return g


def get_maximum_common_substructure(
    top1: mdtraj.Topology, top2: mdtraj.Topology
) -> dict:
    g1 = get_graph(top1)
    g2 = get_graph(top2)
    nm = nx.algorithms.isomorphism.categorical_node_match("element", "")
    em = nx.algorithms.isomorphism.categorical_edge_match("type", "")
    isomag = nx.algorithms.isomorphism.ISMAGS(g1, g2, node_match=nm, edge_match=em)
    lcss = list(isomag.largest_common_subgraph())

    if len(lcss) > 1:
        Warning(
            "More than one largest common substructures found. Returning the first one."
        )

    return lcss[0]
