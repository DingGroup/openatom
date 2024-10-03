import os
import time

def make_psf_from_topology(topology, psf_file_name):
    file_handle = open(psf_file_name, 'w')

    ## start line
    print("PSF CMAP CHEQ XPLOR", file = file_handle)
    print("", file = file_handle)

    ## title
    print("{:>8d} !NTITLE".format(2), file = file_handle)
    print("* Coarse Grained System PSF FILE", file = file_handle)
    user = os.environ['USER']
    print(f"* DATE: {time.asctime()} CREATED BY USER: {user}", file = file_handle)

    ## atoms
    num_atoms = topology.getNumAtoms()
    print("", file = file_handle)
    print("{:>8d} !NATOM".format(num_atoms), file = file_handle)
    for atom in topology.atoms():
        name = atom.name
        atom_index = atom.index + 1
        resname = atom.residue.name
        res_index = atom.residue.index + 1

        segment = atom.residue.chain.id
        print("{:>8d} {:<4s} {:<4d} {:<4s} {:<4s} {:<4s} {:<14.6}{:<14.6}{:>8d}{:14.6}".format(atom_index, segment, res_index, resname, name, name, 0.0, 0.0, 0, 0.0), file = file_handle)

    ## bonds
    num_bonds = topology.getNumBonds()
    print("", file = file_handle)
    print("{:>8d} !NBOND: bonds".format(num_bonds), file = file_handle)

    count = 0
    for bond in topology.bonds():
        atom1 = bond[0].index + 1
        atom2 = bond[1].index + 1
        print("{:>8d}{:>8d}".format(atom1, atom2), file = file_handle, end = '')
        count += 1
        if count == 4:
            print("", file = file_handle)
            count = 0
    print("", file = file_handle)
    
    file_handle.close()