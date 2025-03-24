import openmm
import networkx as nx
import numpy as np
import copy
import multiprocessing as mp
from itertools import chain
try: 
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


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

def topology_to_rdkit(topology: openmm.app.Topology, exclude_labels=None) -> "rdkit.Chem.Mol":
    """Convert an OpenMM topology to an RDKit molecule.
    
    Args:
        topology (openmm.app.Topology): OpenMM topology object.
        
    Returns:
        rdkit.Chem.Mol: RDKit molecule object with OpenMMIndex property on each atom.
    """
    if not HAS_RDKIT:
        return None
    
    exclude_labels = exclude_labels or []
    exclude_atoms = set()
    if exclude_labels:
        for atom in topology.atoms():
            if hasattr(atom, "label") and atom.label in exclude_labels:
                exclude_atoms.add(atom.index)
    mol = Chem.EditableMol(Chem.Mol())
    
    atom_idx_map = {}  
    for atom in topology.atoms():
        element = atom.element.symbol
        if element == 'D':  
            rd_atom = Chem.Atom('H')
            rd_atom.SetIsotope(2)
        else:
            rd_atom = Chem.Atom(element)
        
        if atom.index in exclude_atoms:
            rd_atom.SetAtomicNum(114)  # dummy atom

        charge = 0
        if hasattr(atom, 'formalCharge') and atom.formalCharge is not None:
            charge = atom.formalCharge
        rd_atom.SetFormalCharge(charge)
        

        idx = mol.AddAtom(rd_atom)
        atom_idx_map[atom.index] = idx
    

    for bond in topology.bonds():
        i = atom_idx_map[bond.atom1.index]
        j = atom_idx_map[bond.atom2.index]
        
        bond_type = Chem.BondType.SINGLE
        if hasattr(bond, 'type'):
            if bond.type == 'double':
                bond_type = Chem.BondType.DOUBLE
            elif bond.type == 'triple':
                bond_type = Chem.BondType.TRIPLE
            elif bond.type == 'aromatic':
                bond_type = Chem.BondType.AROMATIC
        
        mol.AddBond(i, j, bond_type)
    

    rdkit_mol = mol.GetMol()
    

    for omm_idx, rd_idx in atom_idx_map.items():
        rdkit_mol.GetAtomWithIdx(rd_idx).SetProp('OpenMMIndex', str(omm_idx))
        rdkit_mol.GetAtomWithIdx(rd_idx).SetProp('Excluded', 
                                              'True' if omm_idx in exclude_atoms else 'False')
    
    try:
        Chem.SanitizeMol(rdkit_mol)
        return rdkit_mol
    except Exception as e:
        import warnings
        warnings.warn(f"RDKit sanitization failed: {str(e)}. Returning unsanitized molecule.")
        return rdkit_mol


def compute_mcs_RDKit(top1: openmm.app.Topology, top2: openmm.app.Topology, **kwargs) -> dict:
    """Compute the maximum common substructure between two topologies using RDKit.

    Args:
        top1 (openmm.app.Topology): OpenMM topology object of the first molecule.
        top2 (openmm.app.Topology): OpenMM topology object of the second molecule.
        **kwargs: Keyword arguments to be passed to the RDKit MCS function.
            

    Returns:
        dict: A dictionary where the keys are the atom indices of the first molecule and
        the values are the atom indices of the second molecule that are in the common
        substructure.

    """
    if not HAS_RDKIT:
        import warnings
        warnings.warn("RDKit is not installed. Please install RDKit to use this function.")
        return {}
    
    # parameters for the MCS algorithm
    default_params = {
        'atomCompare': rdFMCS.AtomCompare.CompareIsotopes,
        'bondCompare': rdFMCS.BondCompare.CompareOrder,
        'matchValences': False,
        'ringMatchesRingOnly': True,
        'completeRingsOnly': True,
        'timeout': 60  # seconds
    }
    mcs_params = default_params.copy()
    mcs_params.update(kwargs)
    mol1 = topology_to_rdkit(top1)
    mol2 = topology_to_rdkit(top2)
    
    if mol1 is None or mol2 is None:
        return {}
    
    # Use RDKit to find MCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2], **mcs_params)
    
    if mcs_result.numAtoms == 0:
        return {}
    
    # Get the MCS as a molecule
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    # Find atom mappings
    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)
    
    if not match1 or not match2:
        return {}
    
    # Create mapping between original topologies
    mapping = {}
    for i, j in zip(match1, match2):
        openmmidx1 = int(mol1.GetAtomWithIdx(i).GetProp('OpenMMIndex'))
        openmmidx2 = int(mol2.GetAtomWithIdx(j).GetProp('OpenMMIndex'))
        mapping[openmmidx1] = openmmidx2
    graph = nx.Graph()
    for i, j in mapping.items():
        graph.add_node(i)
    for i, j in mapping.items():
        for neighbor in list(graph.nodes):
            if i != neighbor and any(b.atom1.index == i and b.atom2.index == neighbor 
                                      for b in top1.bonds()):
                graph.add_edge(i, neighbor)
    
    components = list(nx.connected_components(graph))
    if not components:
        return {}
    
    largest_component = max(components, key=len)
    connected_mapping = {i: mapping[i] for i in largest_component}
    
    return connected_mapping
