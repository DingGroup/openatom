import networkx as nx
import openmm as omm
import xml.etree.ElementTree as ET
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from collections import defaultdict


def align_coordinates(
    x1: np.ndarray,
    x2: np.ndarray,
    atoms1: list[int],
    atoms2: list[int],
) -> np.ndarray:
    x1_center = x1[atoms1].mean(axis=0)
    x2_center = x2[atoms2].mean(axis=0)

    x2 = x2 - x2_center
    r = Rotation.align_vectors(x1[atoms1] - x1_center, x2[atoms2])[0]

    return r.apply(x2) + x1_center


def get_graph(topology: omm.app.Topology) -> nx.Graph:
    g = nx.Graph()
    for bond in topology.bonds():
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
    top1: omm.app.Topology, top2: omm.app.Topology
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
    ligands: list[ET.Element],
    common_particles: list[list[int]],
    environment: ET.Element,
):
    """Merge particles from the environment and ligands into the target system.

    The common particles are added from the first ligand.
    Soft-core particles are added from all ligands.
    All particles from the environment are added.

    As particles from ligands and environment are added, their indices in the target system
    are recorded in the "idx" attribute of the corresponding particle. This is useful for
    later merging of forces.

    Args:
        environment (ET.Element): XML element of the environment.
            It should be the root of the environment XML tree.
        ligands (list[ET.Element]): List of XML elements of ligands.
            Each element should be the root of the ligand XML tree.
        common_particles (list[list[int]]): List of list of indices of common particles in
            each ligand. The order of the list should be the same as the order of ligands.
            Moverover, the order of indices in each list should match each other, meaning that
            the ith index in each list should correspond to the same common particle.
    """

    particles = ET.Element("Particles")

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

    return particles


def merge_constraints(ligands: list[ET.Element], environment: ET.Element):
    constraints = ET.Element("Constraints")
    for idx_lig, lig in enumerate(ligands):
        for c in lig.iterfind("./Constraints/Constraint"):
            i1, i2 = int(c.get("p1")), int(c.get("p2"))
            j1, j2 = _get_idx(lig, [i1, i2])
            if idx_lig == 0:
                ET.SubElement(
                    constraints,
                    "Constraint",
                    {
                        "d": c.get("d"),
                        "p1": str(j1),
                        "p2": str(j2),
                    },
                )
            else:
                if (
                    lig.find("./Particles")[i1].get("class") == "soft-core"
                    or lig.find("./Particles")[i2].get("class") == "soft-core"
                ):
                    ET.SubElement(
                        constraints,
                        "Constraint",
                        {
                            "d": c.get("d"),
                            "p1": str(j1),
                            "p2": str(j2),
                        },
                    )

    for c in environment.iterfind("./Constraints/Constraint"):
        i1, i2 = int(c.get("p1")), int(c.get("p2"))
        j1, j2 = _get_idx(environment, [i1, i2])
        ET.SubElement(
            constraints,
            "Constraint",
            {
                "d": c.get("d"),
                "p1": str(j1),
                "p2": str(j2),
            },
        )
    return constraints


def _get_idx(root: ET.Element, idxs: list[int]) -> list[int]:
    return [int(root.find("./Particles")[i].get("idx")) for i in idxs]


def merge_harmonic_bonds(
    ligands: list[ET.Element],
    scaling_factors: list[(float, float)],
    environment: ET.Element,
):
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

    return force


def merge_harmonic_angles(
    ligands: list[ET.Element],
    scaling_factors: list[(float, float)],
    environment: ET.Element,
):
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

    return force


def merge_periodic_torsions(
    ligands: list[ET.Element],
    scaling_factors: list[(float, float)],
    environment: ET.Element,
):
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

    lig_fs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "PeriodicTorsionForce":
                lig_fs.append(f)
    assert len(lig_fs) == len(ligands)

    env_ptf = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "PeriodicTorsionForce":
            env_ptf = f

    for (_, svdw), lig_f, lig in zip(scaling_factors, lig_fs, ligands):
        for t in lig_f.iterfind("./Torsions/Torsion"):
            i1, i2, i3, i4 = (
                int(t.get("p1")),
                int(t.get("p2")),
                int(t.get("p3")),
                int(t.get("p4")),
            )
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
        i1, i2, i3, i4 = (
            int(t.get("p1")),
            int(t.get("p2")),
            int(t.get("p3")),
            int(t.get("p4")),
        )
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
    scaling_factors: list,
    environment: ET.Element,
):
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            attrib = f.attrib

    force = ET.Element("Force", attrib)
    ET.SubElement(force, "GlobalParameters")
    ET.SubElement(force, "ParticleOffsets")
    ET.SubElement(force, "ExceptionOffsets")

    particles = ET.SubElement(force, "Particles")

    for i, lig in enumerate(ligands):
        for p in lig.iterfind("./Particles/Particle"):
            if i == 0:
                ET.SubElement(
                    particles, "Particle", {"eps": "0.0", "q": "0.0", "sig": "0.0"}
                )
            if i > 0 and p.get("class") == "soft-core":
                ET.SubElement(
                    particles,
                    "Particle",
                    {
                        "eps": "0.0",
                        "q": "0.0",
                        "sig": "0.0",
                    },
                )

    for p in environment.iterfind("./Particles/Particle"):
        ET.SubElement(particles, "Particle", {"eps": "0.0", "q": "0.0", "sig": "0.0"})

    lig_fs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                lig_fs.append(f)
    assert len(lig_fs) == len(ligands)

    env_f = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            env_f = f

    for (selec, svdw), lig_f, lig in zip(scaling_factors, lig_fs, ligands):
        for i, p in enumerate(lig_f.iterfind("./Particles/Particle")):
            j = int(lig.find("./Particles")[i].get("idx"))

            ## if the ligand particle is a common particle, scale eps, sig and q using svdw
            if lig.find("./Particles")[i].get("class") == "common":
                eps = float(p.get("eps")) * svdw
                q = float(p.get("q")) * svdw
                sigma = float(p.get("sig")) * svdw

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
                particles[j].set("q", str(q * selec))

                ## set the charge of the attached common particle
                k = lig.find("./Particles")[i].get("attach_idx")
                j = int(lig.find("./Particles")[int(k)].get("idx"))
                current_q = float(particles[j].get("q"))
                particles[j].set("q", str(current_q + (1 - selec) * q * svdw))

    for i, p in enumerate(env_f.iterfind("./Particles/Particle")):
        j = int(environment.find("./Particles")[i].get("idx"))
        particles[j].set("eps", p.get("eps"))
        particles[j].set("sig", p.get("sig"))
        particles[j].set("q", p.get("q"))

    ## exceptions
    exception_dict = defaultdict(lambda: {"eps": 0, "q": 0, "sig": 0})
    for (selec, svdw), lig_f, lig in zip(scaling_factors, lig_fs, ligands):
        for e in lig_f.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            j1, j2 = _get_idx(lig, [i1, i2])
            j1, j2 = sorted([j1, j2])

            if (
                lig.find("./Particles")[i1].get("class") == "common"
                and lig.find("./Particles")[i2].get("class") == "common"
            ):
                exception_dict[(j1, j2)]["eps"] += float(e.get("eps")) * svdw
                exception_dict[(j1, j2)]["q"] += float(e.get("q")) * svdw
                exception_dict[(j1, j2)]["sig"] += float(e.get("sig")) * svdw

            elif (
                lig.find("./Particles")[i1].get("class") == "soft-core"
                and lig.find("./Particles")[i2].get("class") == "soft-core"
            ):
                exception_dict[(j1, j2)]["eps"] = float(e.get("eps"))
                exception_dict[(j1, j2)]["q"] = float(e.get("q"))
                exception_dict[(j1, j2)]["sig"] = float(e.get("sig"))

            else:
                exception_dict[(j1, j2)]["eps"] = float(e.get("eps")) * svdw
                exception_dict[(j1, j2)]["q"] = float(e.get("q")) * svdw
                exception_dict[(j1, j2)]["sig"] = float(e.get("sig"))

    for e in env_f.iterfind("./Exceptions/Exception"):
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
    scaling_factors: list,
    environment: ET.Element,
):
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

    for i, lig in enumerate(ligands):
        for p in lig.iterfind("./Particles/Particle"):
            if i == 0:
                ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})
            if i > 0 and p.get("class") == "soft-core":
                ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})
    for p in environment.iterfind("./Particles/Particle"):
        ET.SubElement(particles, "Particle", {"param1": "0", "param2": "0"})

    lig_fs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                lig_fs.append(f)
    assert len(lig_fs) == len(ligands)

    env_f = None
    for f in environment.iterfind("./Forces/Force"):
        if f.get("type") == "NonbondedForce":
            env_f = f

    for idx_lig, ((selec, svdw), lig_f, lig) in enumerate(
        zip(scaling_factors, lig_fs, ligands)
    ):
        for i, p in enumerate(lig_f.iterfind("./Particles/Particle")):
            j = int(lig.find("./Particles")[i].get("idx"))
            eps = float(p.get("eps"))
            sigma = float(p.get("sig"))
            if lig.find("./Particles")[i].get("class") == "common":
                current_eps = float(particles[j].get("param1"))
                current_sigma = float(particles[j].get("param2"))

                particles[j].set("param1", str(current_eps + eps * svdw))
                particles[j].set("param2", str(current_sigma + sigma * svdw))
                particles[j].set("param3", "0")
                particles[j].set("param4", "0")

            else:
                particles[j].set("param1", str(eps))
                particles[j].set("param2", str(sigma))
                particles[j].set("param3", str(svdw))
                particles[j].set("param4", str(idx_lig + 1))

    for i, p in enumerate(env_f.iterfind("./Particles/Particle")):
        j = int(environment.find("./Particles")[i].get("idx"))
        particles[j].set("param1", p.get("eps"))
        particles[j].set("param2", p.get("sig"))
        particles[j].set("param3", "0")
        particles[j].set("param4", "0")

    interaction_groups = ET.SubElement(force, "InteractionGroups")
    interaction_group = ET.SubElement(interaction_groups, "InteractionGroup")
    set1 = ET.SubElement(interaction_group, "Set1")
    set2 = ET.SubElement(interaction_group, "Set2")
    for i, lig in enumerate(ligands):
        for p in lig.iterfind("./Particles/Particle"):
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
    lig_fs = []
    for lig in ligands:
        for f in lig.iterfind("./Forces/Force"):
            if f.get("type") == "NonbondedForce":
                lig_fs.append(f)
    assert len(lig_fs) == len(ligands)

    for lig_f, lig in zip(lig_fs, ligands):
        for e in lig_f.iterfind("./Exceptions/Exception"):
            i1, i2 = int(e.get("p1")), int(e.get("p2"))
            if (
                lig.find("./Particles")[i1].get("class") == "common"
                and lig.find("./Particles")[i2].get("class") == "common"
            ):
                continue
            j1, j2 = _get_idx(lig, [i1, i2])
            ET.SubElement(exclusions, "Exclusion", {"p1": str(j1), "p2": str(j2)})

    return force
