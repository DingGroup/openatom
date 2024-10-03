import openmm
import networkx as nx
import numpy as np
import copy
import multiprocessing as mp
from itertools import chain


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

        ## add atom1 and atom2 as nodes, with element as the node attribute
        if symbol1 == "H":
            g.add_node(atom1.index, element=symbol2 + symbol1)
        else:
            g.add_node(atom1.index, element=symbol1)

        if symbol2 == "H":
            g.add_node(atom2.index, element=symbol1 + symbol2)
        else:
            g.add_node(atom2.index, element=symbol2)

        ## add the bond between atom1 and atom2 as an edge, with its type as the edge attribute
        g.add_edge(atom1.index, atom2.index, type=bond.type)

        ## we sometimes want to associate extra information with the atoms besides the element
        ## to do this, we can add a label attribute to the atom in the input topology
        if hasattr(atom1, "label"):
            g.add_node(atom1.index, label=atom1.label)
        if hasattr(atom2, "label"):
            g.add_node(atom2.index, label=atom2.label)

    ## add the degree of each node as an attribute
    nx.set_node_attributes(g, dict(g.degree), "degree")

    return g


def compute_mapping(ga: nx.Graph, gb: nx.Graph, source: int) -> dict:
    """Compute the subgraph isomorphism between ga and gb starting from source node in gb.

    ga is assumed to be the bigger graph and gb is the smaller graph.

    Args:
        ga (nx.Graph): NetworkX graph of the bigger molecule.
        gb (nx.Graph): NetworkX graph of the smaller molecule.
        source (int): The node in gb to start the subgraph isomorphism from.

    Returns:
        dict: A dictionary where the keys are the node indices of the bigger molecule and
        the values are the node indices of the smaller molecule that are in the common
        substructure

    """
    gb_copy = copy.deepcopy(gb)

    #### start with the whole graph of gb, we will remove nodes from gb one node at a time
    #### until that the remaining subgraph of gb is isomorphic to a subgraph of ga

    ## we will remove nodes from gb in a breadth-first search order starting from source
    ## the source node is assume to be a node with degree 1, i.e. at atom with only one bond
    bfs_successors = list(nx.bfs_successors(gb_copy, source))
    nodes = list(chain(*[v for _, v in bfs_successors]))
    nodes.insert(0, source)

    ## node matching function
    nm = nx.algorithms.isomorphism.categorical_node_match(
        ["element", "degree", "label"], default=["", "", ""]
    )

    ## edge matching function
    em = nx.algorithms.isomorphism.categorical_edge_match("type", default="")

    
    core = {}
    for n in nodes:
        gb_copy.remove_node(n)
        gm = nx.algorithms.isomorphism.GraphMatcher(
            ga, gb_copy, node_match=nm, edge_match=em
        )
        if gm.subgraph_is_isomorphic():
            core = gm.mapping
            break

    ## if we cannot find a subgraph of gb that is isomorphic to a subgraph of ga
    ## by removing nodes from gb, we give up at this point
    if len(core) == 0:
        return {}

    ## starting from the subgraph of gb discovered above, we will grow the subgraph
    ## by adding one node at a time and check if the subgraph is isomorphic to a subgraph of ga
    source = list(core.values())[0]
    subnodes = [source]
    bfs_successors = list(nx.bfs_successors(gb, source))
    nodes = list(chain(*[v for _, v in bfs_successors]))
    for n in nodes:
        gm = nx.algorithms.isomorphism.GraphMatcher(
            ga, nx.subgraph(gb, subnodes + [n]), node_match=nm, edge_match=em
        )
        if gm.subgraph_is_isomorphic():
            subnodes.append(n)

    gm = nx.algorithms.isomorphism.GraphMatcher(
        ga, nx.subgraph(gb, subnodes), node_match=nm, edge_match=em
    )

    gm.subgraph_is_isomorphic()

    return gm.mapping


def compute_mcs_VF2(
    top1: openmm.app.Topology, top2: openmm.app.Topology, timeout=10
) -> dict:
    """Compute the maximum common substructure between two topologies.

    Each topology is converted to a NetworkX graph using the make_graph function and
    the maximum common substructure is computed using the VF2 algorithm implemented in NetworkX
    based on the NetworkX graphs of the topologies.

    Args:
        top1 (openmm.app.Topology): OpenMM topology object of the first molecule.
        top2 (openmm.app.Topology): OpenMM topology object of the second molecule.
        timeout (int): The maximum time in seconds to wait for the computation to finish.
    """

    if top1.getNumAtoms() >= top2.getNumAtoms():
        top_large, top_small = top1, top2
        key = "first"
    else:
        top_large, top_small = top2, top1
        key = "second"

    gl = make_graph(top_large)
    gs = make_graph(top_small)

    nodes_with_one_bond = [n for n in gs.nodes if gs.degree(n) == 1]

    if mp.cpu_count() > 16 and len(nodes_with_one_bond) > 16:
        num_processes = 16
    else:
        num_processes = min(mp.cpu_count(), len(nodes_with_one_bond))

    mappings = []
    with mp.Pool(num_processes) as pool:
        futures = [
            pool.apply_async(compute_mapping, args=(gl, gs, n))
            for n in nodes_with_one_bond
        ]
        pool.close()
        for future in futures:
            try:
                mapping = future.get(timeout=timeout)
                mappings.append(mapping)
            except mp.TimeoutError:
                None

    M = max([len(m) for m in mappings])
    if M == 0:
        return {}

    mappings = [m for m in mappings if len(m) == M]
    mapping = min(mappings, key=lambda x: sum([abs(k - v) for k, v in x.items()]))

    subgraph_large = gl.subgraph(list(mapping.keys()))
    components = list(nx.connected_components(subgraph_large))
    len_components = [len(c) for c in components]
    largest_component = components[np.argmax(len_components)]

    connected_lcs = {i: mapping[i] for i in largest_component}

    if key == "first":
        return connected_lcs
    else:
        return {v: k for k, v in connected_lcs.items()}


def compute_mcs_ISMAGS(top1: openmm.app.Topology, top2: openmm.app.Topology) -> dict:
    """Compute the maximum common substructure between two topologies.

    Each topology is converted to a NetworkX graph using the make_graph function and
    the maximum common substructure is computed using the ISMAGS algorithm implemented in NetworkX
    based on the NetworkX graphs of the topologies.

    Note: This function is slower than compute_mcs_VF2.

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
    nm = nx.algorithms.isomorphism.categorical_node_match(
        ["element", "degree"], ["", ""]
    )
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
