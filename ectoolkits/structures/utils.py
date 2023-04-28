import numpy as np


from ase.build import molecule
from ase.geometry import get_distances
from ase import Atoms

from ase.neighborlist import neighbor_list

from ecflow_bandalign.tools.slab import divide_terminal_idxs, rerange_terminal_idxs


def get_uniform_idxs(nsample, num=1):
    import numpy as np
    assert nsample>=num
    indices = np.linspace(start=0, stop=nsample, num=num, endpoint=False, dtype=int)

    return indices

def rotate_water_to_position(water, position,):
    
    assert all(water.symbols =="OH2")
    
    water = water.copy()
    v1 = get_distances(water[0].position, water[1].position, cell=water.cell, pbc=True)[0][0][0]
    orient = get_distances(water[0].position, position, cell=water.cell, pbc=True)[0][0][0]
    water.rotate(v1, orient, water[0].position)
    
    return water

def change_molecule_bond_length(mol, start_idx, end_idx, bond_length):
    
    mol = mol.copy()
    oh_length = mol.get_distance(start_idx, end_idx)
    scale_factor = bond_length / oh_length
    mol.positions[1] = mol.positions[0] + scale_factor * (mol.positions[1] - mol.positions[0])

    return mol

def search_coord_number(atoms, element, coord_element, cutoff=2.6):
    
    #给定atoms 和中心元素 以及配位元素和最大配位数，输出整体元素的配位情况

    structure = atoms
    from ase.neighborlist import neighbor_list 
    
    #TODO 原来还是用的ase的neighbor_list 很简陋 需要改进，没有考虑mic
    coord_num = np.bincount(neighbor_list('i', structure, {(element, coord_element): cutoff}))
    element_idx = np.array([atom.index for atom in structure if atom.symbol == element])
    num_coord = coord_num[element_idx]
    
    site_dict = {}
    
    coord_species = list(set(num_coord)) #得到 中心原子的配位数的种类
    #为所有配位数的可能情况创建空字典
    for num in coord_species:
        site_dict[num] = np.array([],dtype=int)
    
    for idx in element_idx:
        site_dict[coord_num[idx]] = np.append(site_dict[coord_num[idx]], idx)
    
    return site_dict