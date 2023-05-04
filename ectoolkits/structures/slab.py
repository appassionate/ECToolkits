from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.io import read, write
from ase.build import surface
from ..utils.math import get_plane_eq
import numpy as np
import os
import shutil
from typing import Tuple, List
from MDAnalysis.lib.distances import minimize_vectors

from .utils import search_coord_number, get_uniform_idxs
from ase.geometry import get_distances
from ase.build import molecule


class Slab(Atoms):
    """
        Object inherent from Atoms object in ASE.
        Add method for vacuum slab model
    Args:
        Atoms (_type_): Atoms object int ASE
    """
    
    #more special 
    indices:str = None
    primitive:Atoms = None
    n_layers:int = None
    lateral_repeat:list = None
    
    def copy(self):
        
        #override Atoms copy method to support more attri in Slab Object
        
        #attention: some object may not deepcopy
        slab = Atoms.copy(self)

        slab.indices = self.indices
        slab.primitive = self.primitive
        slab.n_layers = self.n_layers
        slab.lateral_repeat = self.lateral_repeat

        return slab
    
    def _reset_slab_attri(self, indices=None, primitive=None, n_layers=None, lateral_repeat=None):
        
        if indices:
            self.indices = indices
        if primitive:
            self.primitive = primitive
        if n_layers:
            self.n_layers = n_layers
        if lateral_repeat:
            self.lateral_repeat = lateral_repeat
        
    
    def get_cus(self, input_idx, coord_num, cutoff):
        """
        function to get atom index of coordinate unsaturated sites.
        slab: Atoms object, the slab model
        input_idx: the index of the atom you want get the coordination number
        coord_num: coordination number for coordinate unsaturated sites, the number must be less then the full coordination
        cutoff: the cutoff radius defining coordination. something like: {('Ti', 'O'): 2.2}
        return: the index for cus atoms
        """
        coord_num_list = np.bincount(neighbor_list('i', self, cutoff))[input_idx]
        target_idx = input_idx[coord_num_list == coord_num]
        return target_idx
    
    def get_neighbor_list(self, idx: int, cutoff: dict) -> list:        
        """
            provided that atom index and return its neighbor in list
        Args:
            idx (int): atom index
            cutoff (dict): cutoff for neighbor pair

        Returns:
            list: list of atom indices
        
        Examples:
            get_neighbor_list(16, {("O", "H"): 1.4})
        """        
        i, j = neighbor_list('ij', self, cutoff=cutoff)
        return j[i==idx]

    def find_element_idx_list(self, element: str) -> list:
        """
        find atom index provided that element symbol

        _extended_summary_

        Args:
            element (str): element symbol

        Returns:
            list: list of atom indices
        """            
        cs = self.get_chemical_symbols()
        cs = np.array(cs)
        idx_list = np.where(cs==element)[0]
        return list(idx_list)

    def find_surf_idx(self, 
                      element:str=None, 
                      tolerance:float=0.1, 
                      dsur:str='up',
                      check_cross_boundary=False,
                      trans_z_dist = 5
                      ) -> list:
        """
            find atom indexs at surface

        _extended_summary_

        Args:
            element (str): element symbol
            tolerance (float, optional): tolerance for define a layer. Defaults to 0.1.
            dsur (str, optional): direction of surface, 'up' or 'dw'. for a vacuum-slab model, 
            you have up surface and down surface. Defaults to 'up'.

        Returns:
            list: list of atom indices
        """ 
        tmp_stc = self.copy()  
        if check_cross_boundary:
            while tmp_stc.is_cross_z_boundary(element=element):
                print(f"The slab part is cross z boundary, tranlate {trans_z_dist:3.3f} A!")
                tmp_stc.translate([0,0,trans_z_dist])
                tmp_stc.wrap()

        if element:
            idx_list = tmp_stc.find_element_idx_list(element)
            z_list = tmp_stc[idx_list].get_positions().T[2]
        else: 
            z_list = tmp_stc.get_positions().T[2]
        if dsur == 'up':
            z = z_list.max()
        elif dsur == 'dw':
            z = z_list.min()

        zmin = z-tolerance
        zmax = z+tolerance
        idx_list = tmp_stc.find_idx_from_range(zmin=zmin, zmax=zmax, element=element)
    
        return idx_list

    def del_surf_layer(self, element: str =None, tolerance=0.1, dsur='up', check_cross_boundary=False):
        """ delete the layer atoms,

        _extended_summary_

        Args:
            element (str, optional): _description_. Defaults to None.
            tolerance (float, optional): _description_. Defaults to 0.1.
            dsur (str, optional): _description_. Defaults to 'up'.

        Returns:
            _type_: _description_
        """        

        del_list = self.find_surf_idx(element=element, 
                                      tolerance=tolerance, 
                                      dsur=dsur, 
                                      check_cross_boundary=check_cross_boundary
                                      )
        
        tmp = self.copy()
        del tmp[del_list]
        return tmp

    def find_idx_from_range(self, zmin:int, zmax:int, element: str =None) -> list:
        """_summary_

        _extended_summary_

        Args:
            zmin (int): minimum in z 
            zmax (int): maximum in z
            element (str, optional): element symbol, None means all atoms. Defaults to None.

        Returns:
            list: list of atom indices
        """        
        idx_list = []
        if element:
            for atom in self:
                if atom.symbol == element:
                    if (atom.position[2] < zmax) and (atom.position[2] > zmin):
                        idx_list.append(atom.index)
        else:
            for atom in self:
                if (atom.position[2] < zmax) and (atom.position[2] > zmin):
                    idx_list.append(atom.index)       
        return idx_list

    def del_from_range(self, zmin: int, zmax: int, element: str =None) -> Atoms:
        """_summary_

        _extended_summary_

        Args:
            zmin (int): _description_
            zmax (int): _description_
            element (str, optional): _description_. Defaults to None.

        Returns:
            Atoms: _description_
        """           
        tmp = self.copy()
        del_idx_list = self.find_idx_from_range(zmin=zmin, zmax=zmax, element=element)

        del tmp[del_idx_list]
        
        return tmp
    
    def add_adsorbate(self, 
                  ad_site_idx:int, 
                  vertical_dist:float, 
                  adsorbate:Atoms, 
                  contact_atom_idx:int=0,
                  lateral_shift:Tuple[float]=(0,0),
                  ):

        tmp_stc = self.copy()
        site_pos = tmp_stc[ad_site_idx].position.copy()
        tmp_adsorbate = adsorbate.copy()
        # refer the positions of adsorbate to the contact_atom
        contact_atom_pos = tmp_adsorbate[contact_atom_idx].position.copy()
        tmp_adsorbate.translate(-contact_atom_pos)
        # move the adsorbate to target position
        target_pos = site_pos+np.array([lateral_shift[0], lateral_shift[1], vertical_dist])
        tmp_adsorbate.translate(target_pos)
        tmp_stc.extend(tmp_adsorbate)
        return tmp_stc
    
    def add_adsorbates(self,
                    ad_site_idx_list:List[int], 
                    vertical_dist:float, 
                    adsorbate:Atoms, 
                    contact_atom_idx:int=0,
                    lateral_shift:Tuple[float]=(0,0),
                    ):
        tmp_stc = self.copy()
        for ad_site_idx in ad_site_idx_list:
            tmp_stc = tmp_stc.add_adsorbate(ad_site_idx=ad_site_idx,
                                      vertical_dist=vertical_dist,
                                      adsorbate=adsorbate, 
                                      contact_atom_idx=contact_atom_idx,
                                      lateral_shift=lateral_shift,
                                      )
        return tmp_stc
        

    def generate_interface(self, 
                           water_box_len: float, 
                           top_surface_idx: List[int], 
                           bottom_surface_idx: List[int]
                           ):
        """merge slab model and water box together

        Args:
            water_box_len:
            top_surface_idx:
            bottom_surface_idx:

        Returns:
            tmp:
        """
        # find the water box
        if os.path.exists("gen_water/watbox.xyz"):
            water_box = read("gen_water/watbox.xyz")
            print("Water Box Found")
        else:
            print("Water Box Not Found")
            raise FileNotFoundError('Water box not found, please install packmol')

        tmp = self.copy()
        cell_z = tmp.get_cell()[2][2]
        # shift the water in z directions (to add in slab model)
        tmp_water_positions = water_box.get_positions()
        for i in range(len(tmp_water_positions)):
            tmp_water_positions[i] += [0, 0, cell_z + 0.5]
        water_box.set_positions(tmp_water_positions)
        # add the water box to slab model
        tmp.extend(water_box)
        # modify the z length
        tmp.set_cell(tmp.get_cell() + [[0, 0, 0], [0, 0, 0], [0, 0, water_box_len + 1]])
        # shift the water center to box center
        top_surface_z = tmp[top_surface_idx].get_positions().T[2].mean()
        bottom_surface_z = tmp[bottom_surface_idx].get_positions().T[2].mean()
        slab_center_z = 0.5 * (top_surface_z + bottom_surface_z)
        tmp.translate([0, 0, -slab_center_z])
        tmp.set_pbc([False, False, True])
        tmp.wrap()
        print("Merge Water and Slab Box Finished")
        return tmp
    
    def generate_water_box(self, water_box_len):
        """function to generate water box
        x and y length is from self length
        Args:
            water_box_len:

        Returns:

        """
        cell = self.get_cell()
        cell_a = cell[0]
        cell_b = cell[1]
        header = "-"
        print(header * 50)
        print("Now Generate Water Box")
        space_per_water = 9.86 ** 3 / 32
        wat_num = (np.linalg.norm(np.cross(cell_a,cell_b)) * water_box_len) / space_per_water
        wat_num = int(wat_num)
        #print("Read Cell X: {0:03f} A".format(cell_a))
        #print("Read Cell Y: {0:03f} A".format(cell_b))
        print("Read Water Box Length: {0:03f} A".format(water_box_len))
        print("Predict Water Number: {0}".format(wat_num))
        
        n_vec_a, d1_a, d2_a, n_vec_b, d1_b, d2_b = get_plane_eq(cell_a, cell_b)
        print("Calculate Plane Equation")

        if os.path.exists('gen_water'):
            print("found gen_water direcotry, now remove it")
            shutil.rmtree('gen_water')
        print("Generate New Directory: gen_water")
        os.mkdir('gen_water')
        print("Generate Packmol Input: gen_wat_box.inp")
        with open(os.path.join("gen_water", "gen_wat_box.inp"), 'w') as f:
            txt = "#packmol input generate by python"
            txt += "\n"
            txt += "tolerance 2.0\n"
            txt += "filetype xyz\n"
            txt += "output watbox.xyz"
            txt += "\n"
            txt += "structure water.xyz\n"
            txt += "  number {0}\n".format(int(wat_num))
            txt += "  above plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(n_vec_a[0], n_vec_a[1], n_vec_a[2], d1_a+0.5)
            txt += "  below plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(n_vec_a[0], n_vec_a[1], n_vec_a[2], d2_a-0.5)
            txt += "  above plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(n_vec_b[0], n_vec_b[1], n_vec_b[2], d1_b+0.5)
            txt += "  below plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(n_vec_b[0], n_vec_b[1], n_vec_b[2], d2_b-0.5)
            txt += "  above plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(0, 0, 1.0, 0.)
            txt += "  below plane {0:6.4f} {1:6.4f} {2:6.4f} {3:6.4f}\n".format(0, 0, 1.0, water_box_len)
            txt += "end structure\n"
            f.write(txt)
        print("Generate A Water Molecule: water.xyz")
        with open(os.path.join("gen_water", "water.xyz"), 'w') as f:
            txt = '3\n'
            txt += ' water\n'
            txt += ' H            9.625597       6.787278      12.673000\n'
            txt += ' H            9.625597       8.420323      12.673000\n'
            txt += ' O           10.203012       7.603800      12.673000\n'
            f.write(txt)
        print("Generate Water Box: watbox.xyz")
        os.chdir("./gen_water")
        os.system("packmol < gen_wat_box.inp")
        os.chdir("../")
        print("Generate Water Box Finished")

    def remove_cell_vacuum(self, adopt_space=2):
        """remove the vacuum of z direction
         cell z must be perpendicular to xy plane
        """
        tmp = self.copy()
        z_list = tmp.get_positions().T[2]
        slab_length = z_list.max() - z_list.min()
        slab_length += 2
        a = tmp.get_cell()[0]
        b = tmp.get_cell()[1]
        c = [0, 0, slab_length]
        tmp.set_cell([a, b, c])
        tmp.center()
        return tmp

    def is_cross_z_boundary(
        self,
        element: str = None
        ):
        # check if slab cross z boundary
        if element:
            M_idx_list = self.find_element_idx_list(element=element)
        else:
            M_idx_list = list(range(len(self)))

        cellpar = self.cell.cellpar()

        coords = self[M_idx_list].get_positions()
        coords_z = coords[:, 2]

        coord_z_max = coords[coords_z.argmax()]
        coord_z_min = coords[coords_z.argmin()]
        vec_raw = coord_z_max - coord_z_min

        vec_minimized = minimize_vectors(vectors=vec_raw, box=cellpar)
        
        #print(vec_minimized[2], vec_raw[2])
        if np.isclose(vec_minimized[2], vec_raw[2], atol=1e-5, rtol=0):
            return False
        else:
            return True
    

class RutileSlab(Slab):
    """
    class atoms used for rutile like(structure) system
    space group: P42/mnm
    Usage:
    rutile = read("Rutile-exp.cif")
    x = RutileType(rutile)
    slab = []
    for i in range(3, 7):
        slab.append(x.get_slab(indices=(1, 1, 0), n_layers=i, lateral_repeat=(2, 4)))
    """

    @property
    def e_metal(self):
        
        eles = list(set(self.symbols))
        eles.remove("O")
        try:
            eles.remove("H")
        except:
            pass
        e_metal = eles.pop()
        return e_metal
    
    @classmethod
    def get_slab(self, primitive, indices: Tuple[int], n_layers, lateral_repeat: Tuple[int]=(2, 4), vacuum=10.0):
        h, k, l = indices
        entry = str(h)+str(k)+str(l)
        method_entry = {
            "110": self.rutile_slab_110,
            "001": self.rutile_slab_001,
            "100": self.rutile_slab_100,
            "101": self.rutile_slab_101,
        }
        
        method = method_entry.get(entry, None)

        try:
            assert method is not None
            slab = method(primitive=primitive, n_layers=n_layers, lateral_repeat=lateral_repeat, vacuum=vacuum)
        except:
            print("Current Miller Index has not implemented yet")
        # if method is None:
        #     raise ValueError("Current Miller Index has not implemented yet")

        return slab

    @classmethod
    def rutile_slab_110(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 110 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell
        
        slab = surface(primitive, (1, 1, 0), n_layers+1, vacuum)
        slab = cls(slab)
        # remove bottom layer
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='up')

        # create the super cell
        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]
        
        slab.indices = "110"
        slab.primitive = primitive
        slab.n_layers = n_layers
        slab.lateral_repeat = lateral_repeat

        return slab

    @classmethod
    def rutile_slab_001(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 2), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 001 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell

        if n_layers%2 == 1:
            slab = surface(primitive, (0, 0, 1), int(n_layers/2)+1, vacuum)
            slab = cls(slab)
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        elif n_layers%2 == 0:
            slab = surface(primitive, (0, 0, 1), int(n_layers/2), vacuum)
            slab = cls(slab)
        
        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]
        
        slab.indices = "001"
        slab.primitive = primitive
        slab.n_layers = n_layers
        slab.lateral_repeat = lateral_repeat

        return slab
    
    @classmethod
    def rutile_slab_100(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 3), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 100 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell    

        if n_layers%2 == 1:
            slab = surface(primitive, (1, 0, 0), int(n_layers/2)+1, vacuum)
            slab = cls(slab)
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
        elif n_layers%2 == 0:
            slab = surface(primitive, (1, 0, 0), int(n_layers/2)+1, vacuum)
            slab = cls(slab)
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]

        slab.indices = "100"
        slab.primitive = primitive
        slab.n_layers = n_layers
        slab.lateral_repeat = lateral_repeat

        return slab

    @classmethod
    def rutile_slab_101(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 2), vacuum=10.0):
        """
        function for create symmetry slab for rutile structure 101 surface
        space group: P42/mnm
        this function is valid only for 6 atoms conventional cell.
        """
        # create six layer and a supercell    
        slab = surface(primitive, (1, 0, 1), n_layers+1, vacuum)
        slab = cls(slab)
        
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='dw')
        slab = slab.del_surf_layer(tolerance=0.1, dsur='up')

        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)
        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]
        
        slab.indices = "101"
        slab.primitive = primitive
        slab.n_layers = n_layers
        slab.lateral_repeat = lateral_repeat
        
        return slab
    
    def divide_terminal_idxs(self, idxs):

        #slab will be create within vacuum
        #we can divde a series of idxs to  
        half_z = self.cell[2][2]/2
        
        upper_idxs = list(filter(lambda x : self[x].position[2] >= half_z, idxs))
        bottom_idxs = list(filter(lambda x : self[x].position[2] < half_z, idxs))
        
        return bottom_idxs, upper_idxs
    
    def rerange_terminal_idxs(self, terminal_idxs, reshape=None, tolerance=0.1):
        # r_num, c_num to reshape the terminals 
        slab = self.copy()
        #tricky: shrink the box to manage the boundary atoms 
        assert tolerance >= 0
        old_cell = slab.cell.cellpar()
        old_cell[0] = old_cell[0] - tolerance
        old_cell[1] = old_cell[1] - tolerance
        slab.set_cell(old_cell)
        slab.pbc = True
        slab.wrap()
        
        terminal_idxs = np.array(terminal_idxs)
        ter_positions = slab[terminal_idxs].positions
        
        reranged = terminal_idxs[np.lexsort(ter_positions.T[:2])] #x，y reorder
        # reordered by x ,y
        
        if reshape:
            r_num = reshape[0]
            c_num = reshape[1]
            reranged = np.reshape(reranged, (r_num, c_num))

        return reranged
    
    def get_rutile_metal_termials(self, cutoff=2.6):
        
        e_metal = self.e_metal
        
        #search the coord 5 terminals 
        # how to manage other coord atoms
        return search_coord_number(self, e_metal, "O", cutoff)[5]
    
    def find_rutile_metal_terminal_Obr(self, m_terminal_idx):
        
        #find the coord 5 terminal O_br idx
        slab = self

        e_metal = self.e_metal
        oxygen_coord_info = search_coord_number(slab, "O", e_metal)
        
        Obr_idxs = oxygen_coord_info[2] # find all O_brs
        dists =slab.get_distances(m_terminal_idx, Obr_idxs, mic=True)
        #select the two nearby O_br
        Obr_idxs_two = Obr_idxs[np.argsort(dists)][:2] 
        
        # get 
        terminal = slab[m_terminal_idx]
        Obr_1 = slab[Obr_idxs_two[0]]
        Obr_2 = slab[Obr_idxs_two[1]]
        v1 = get_distances(terminal.position, Obr_1.position, cell=slab.cell, pbc=True)[0][0][0]
        v2 = get_distances(terminal.position, Obr_2.position, cell=slab.cell, pbc=True)[0][0][0]

        print(Obr_idxs_two)
        orients = [v1[0]*v2[0] < 0 , v1[1]*v2[1] < 0]# two O_br must be two sides around the terminal Atom 
        for i, orients in enumerate(orients):
            if orients:
                _axis = i
                break
        #reorder by the chosen axis
        if v1[_axis] < v2[_axis]:
            return Obr_idxs_two
        else:
            return Obr_idxs_two[::-1]
    
    def get_reranged_terminals_two_sides(self, tolerance=0.1):
        
        terminals = self.get_rutile_metal_termials().tolist()
        bot_idxs, up_idxs  = self.divide_terminal_idxs(terminals)
        bot_idxs, up_idxs = [self.rerange_terminal_idxs(_idxs, tolerance=tolerance) for _idxs in [bot_idxs, up_idxs]]
    
        return (bot_idxs, up_idxs)
    
    # def add_rutile101_oh_symm(self, rank_idx=0, mode="a", tolerance=0.1):
        
    #     bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
    #     rank_idx = min(rank_idx, len(bot_idxs))

    #     tmp = self.copy()
    #     tmp = tmp.add_rutile101_oh(terminal_idx=bot_idxs[rank_idx], side="down",mode=mode)
    #     tmp = tmp.add_rutile101_oh(terminal_idx=up_idxs[rank_idx], side="up", mode=mode)
        
    #     return tmp
    
    # def add_rutile101_oh(self, terminal_idx, side, mode="a"):
        
    #     #TODO: same as splited water
    #     slab = self
    #     assert mode in ("a","b")
    #     #build a tempalte oh
    #     water = molecule("H2O")
    #     water.rotate(90,"x")
    #     water.rotate(90,"y")
    #     water.rotate(-115,"z")
    #     oh = water[:2]
        
    #     O_brs = slab.find_rutile_metal_terminal_Obr(terminal_idx)
    #     O_brs = O_brs if mode=="a" else O_brs[::-1]
    #     _slab = slab.copy()
        
    #     if side=="up":
    #         multi=1
    #     else:
    #         multi=-1
    #         O_brs = O_brs[::-1]
    #         oh.rotate(180,"z")
    #     O_br = O_brs[0] #choose the O_br to ad H atom
        
    #     _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=2*multi , adsorbate=oh)

    #     return _slab
    
    def add_rutile110_split_water_symm(self, rank_idx=4):
        
        #TODO: 临时这样实现 
        from ecflow_bandalign.tools.water import get_adsorb_rutile_struct_water
        eles = list(set(self.symbols))
        eles.remove("O")
        try:
            eles.remove("H")
        except:
            pass
        e_metal = eles.pop()
        
        waters = get_adsorb_rutile_struct_water(self, e_metal, oxide_element="O", chosen_site=rank_idx)
        
        
        return waters
    
    # def add_rutile100_split_water(self, rank_idx, mode=""):
        
    #     pass
    
    # def add_rutile_100_covered_water(self):
    #     pass
    
    def add_rutile110_split_water_in_converage(self, coverage_percent=0.5, ):
        
        nsites = len(self.get_reranged_terminals_two_sides()[0])
        percents = np.array([i/nsites for i in range(0,nsites+1)])
        #print(percents)
        # 选择函数最相近的覆盖度
        _chosen = np.argsort(abs(percents-coverage_percent))
        _percent = percents[_chosen[0]]
        print(f"current adsorb percent: {_percent}")
        n_adsites = list(range(0,nsites+1))[_chosen[0]]
        #TODO: 这里以后可以整理成一个util method
        ad_rank_idxs = get_uniform_idxs(nsites,n_adsites)
        print(ad_rank_idxs)
        #ad_rank_idxs = np.array(ad_rank_idxs)-1
        
        wats=self[:0]
        for _idx in ad_rank_idxs:
            _wats = self.add_rutile110_split_water_symm(rank_idx=_idx)[-6:]
            wats.extend(_wats)
        
        return self+wats
    
    
    def add_rutile101_split_water_symm(self, rank_idx=4, mode="a", tolerance=0.1):
        
        #TODO: 101的方法暂时有问题
        #101 表面有两种不同环境的Sn未满配位Sn原子，需要做区分再加
        #rank_idx = 4的时候是正常的
        
        #rank_idx length come from get_reranged_terminals_two_sides length
        
        # bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        # rank_idx = min(rank_idx, len(bot_idxs))

        tmp = self.copy()
        # tmp = tmp.add_rutile101_split_water(terminal_idx=bot_idxs[rank_idx], side="down",mode=mode)
        # tmp = tmp.add_rutile101_split_water(terminal_idx=up_idxs[rank_idx], side="up", mode=mode)
        
        tmp = tmp.add_rutile101_split_water(terminal_idx=rank_idx, side="down",mode=mode)
        tmp = tmp.add_rutile101_split_water(terminal_idx=rank_idx, side="up", mode=mode)
        
        return tmp

    
    def add_rutile101_split_water(self, rank_idx, side, mode="a", tolerance=0.1):
        
        
        #根据顺序位点 选择要加水的地方
        bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        rank_idx = min(rank_idx, len(bot_idxs))
        if side=="up":
            terminal_idxs = up_idxs
        else:
            terminal_idxs = bot_idxs
        terminal_idx = terminal_idxs[rank_idx]
        
        
        slab = self
        assert mode in ("a","b")
        #build a tempalte oh
        water = molecule("H2O")
        water.rotate(90,"x")
        water.rotate(90,"y")
        water.rotate(-115,"z")
        oh = water[:2]
        
        O_brs = slab.find_rutile_metal_terminal_Obr(terminal_idx)
        O_brs = O_brs if mode=="a" else O_brs[::-1]
        _slab = slab.copy()
        
        if side=="up":
            multi=1
        else:
            multi=-1
            O_brs = O_brs[::-1]
            oh.rotate(180,"z")
        O_br = O_brs[0] #choose the O_br to ad H atom
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=2*multi , adsorbate=oh)
        _slab = _slab.add_adsorbate(O_br, vertical_dist=1*multi, adsorbate=Atoms("H"))
        
        return _slab
    
    def add_rutile101_surface_water_covered(self, height=2.5, mode="a", tolerance=0.1):
    
        bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        ranks = len(bot_idxs)
        
        tmp = self.copy()
        for rank_idx in range(ranks):
            tmp = tmp.add_rutile101_surface_water(terminal_idx=bot_idxs[rank_idx], side="down",mode=mode, height=height)
            tmp = tmp.add_rutile101_surface_water(terminal_idx=up_idxs[rank_idx], side="up", mode=mode, height=height)
        
        return tmp 
    
    
    def add_rutile101_surface_water(self, terminal_idx, side, mode="a", height=2.5):
                
        _slab = self.copy()
        
        water = molecule("H2O")
        water.rotate(90,"x")
        water.rotate(90,"y")
        water.rotate(170,"z")
        if mode == "b":
            water.rotate(205,"z")
        
        if side=="up":
            multi=1
        else:
            multi=-1
            water.rotate(180,"z")
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=height*multi , adsorbate=water)
        
        return _slab
    
    # def add_rutile100_oh_symm(self, rank_idx=0, mode="a", tolerance=0.1):
        
    #     bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
    #     rank_idx = min(rank_idx, len(bot_idxs))

    #     tmp = self.copy()
    #     tmp = tmp.add_rutile100_oh(terminal_idx=bot_idxs[rank_idx], side="down",mode=mode)
    #     tmp = tmp.add_rutile100_oh(terminal_idx=up_idxs[rank_idx], side="up", mode=mode)
        
    #     return tmp
    
    # def add_rutile100_oh(self, terminal_idx, side, mode="a"):
        
    #     slab = self
        
    #     assert mode in ("a","b")
    #     #build a tempalte oh
    #     water = molecule("H2O")
    #     water.rotate(90,"x")
    #     water.rotate(90,"y")
    #     water.rotate(180,"z")
        
    #     if mode == "a":
    #         oh = water[:2]
    #     if mode == "b":
    #         water.rotate(175,"z")
    #         oh = water[[0,2]]
        
    #     #oh = water
    #     print(terminal_idx)
    #     O_brs = slab.find_rutile_metal_terminal_Obr(terminal_idx)
    #     if side=="down":
    #         O_brs=O_brs[::-1]
    #     O_br = O_brs[0] if mode=="a" else O_brs[1]
        
    #     #找到对称的位点 O_far 取负shift
    #     O_idxs = np.where(slab.symbols=="O")[0]
    #     dists =slab.get_distances(terminal_idx, O_idxs, mic=True)
    #     O_idxs_ard = O_idxs[np.argsort(dists)][:5]
    #     shift = np.array([0.,0.,0.])
    #     for idx in O_idxs_ard:
    #         shift -= get_distances(slab.positions[terminal_idx], slab.positions[idx], 
    #                             cell=slab.cell, pbc=slab.pbc)[0][0][0]    
        
    #     _slab = slab.copy()
    #     #_slab.extend(Atoms("N",positions=[_slab[terminal_idx].position+shift]))
    #     if side=="up":
    #         multi=1
    #     else:
    #         multi=-1
    #         oh.rotate(180,"z")
    #     #print(multi) 
    #     _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=shift[2], adsorbate=oh, lateral_shift=shift[:2])
        
    #     return _slab
    
    
    def add_rutile100_split_water_symm(self, rank_idx=0, mode="a", tolerance=0.1):
        #rank_idx length come from get_reranged_terminals_two_sides length
        
        bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        rank_idx = min(rank_idx, len(bot_idxs))
        
        tmp = self.copy()
        #TODO: 此处特例:  rutile100建水 down,mode=a 与up, mode=b才是对称，因为构造的方法是相反的
        # 构造方法有待重构
        tmp = tmp.add_rutile100_split_water(terminal_idx=bot_idxs[rank_idx], side="down",mode="a")
        tmp = tmp.add_rutile100_split_water(terminal_idx=up_idxs[rank_idx], side="up", mode="b")
        
        return tmp
    
    
    def add_rutile100_split_water(self, terminal_idx, side, mode="a"):
        
        slab = self
        
        assert mode in ("a","b")
        #build a tempalte oh
        water = molecule("H2O")
        water.rotate(90,"x")
        water.rotate(90,"y")
        water.rotate(180,"z")
        
        if mode == "a":
            oh = water[:2]
        if mode == "b":
            water.rotate(175,"z")
            oh = water[[0,2]]
        
        #oh = water
        print(terminal_idx)
        O_brs = slab.find_rutile_metal_terminal_Obr(terminal_idx)
        if side=="down":
            O_brs=O_brs[::-1]
        O_br = O_brs[0] if mode=="a" else O_brs[1]
        
        #找到对称的位点 O_far 取负shift
        O_idxs = np.where(slab.symbols=="O")[0]
        dists =slab.get_distances(terminal_idx, O_idxs, mic=True)
        O_idxs_ard = O_idxs[np.argsort(dists)][:5]
        shift = np.array([0.,0.,0.])
        for idx in O_idxs_ard:
            shift -= get_distances(slab.positions[terminal_idx], slab.positions[idx], 
                                cell=slab.cell, pbc=slab.pbc)[0][0][0]    
        
        _slab = slab.copy()
        #_slab.extend(Atoms("N",positions=[_slab[terminal_idx].position+shift]))
        if side=="up":
            multi=1
        else:
            multi=-1
            oh.rotate(180,"z")
        #print(multi) 
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=shift[2], adsorbate=oh, lateral_shift=shift[:2])
        _slab = _slab.add_adsorbate(O_br, vertical_dist=1*multi, adsorbate=Atoms("H"),)
        
        return _slab
    
    
    def add_rutile100_surface_water_covered(self, mode="a" , tolerance=0.1):
    
        bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        ranks = len(bot_idxs)
        
        tmp = self.copy()
        for rank_idx in range(ranks):
            tmp = tmp.add_rutile100_surface_water(terminal_idx=bot_idxs[rank_idx], side="down",mode=mode)
            tmp = tmp.add_rutile100_surface_water(terminal_idx=up_idxs[rank_idx], side="up", mode=mode)
        
        return tmp
    
    
    def add_rutile100_surface_water(self, terminal_idx, side, mode="a"):

        slab = self.copy()
        
        assert mode in ("a","b")
        #build a tempalte h2o
        water = molecule("H2O")
        # water.rotate(90,"x")
        # water.rotate(90,"y")
        water.rotate(180,"x")
        water.rotate(180,"z")
        
        # if mode != "a":
        #     water.rotate(180,"z")
        
        #oh = water
        
        #找到对称的位点 O_far 取负shift
        O_idxs = np.where(slab.symbols=="O")[0]
        dists =slab.get_distances(terminal_idx, O_idxs, mic=True)
        O_idxs_ard = O_idxs[np.argsort(dists)][:5]
        shift = np.array([0.,0.,0.])
        for idx in O_idxs_ard:
            shift -= get_distances(slab.positions[terminal_idx], slab.positions[idx], 
                                cell=slab.cell, pbc=slab.pbc)[0][0][0]    
        _slab = slab.copy()
        #_slab.extend(Atoms("N",positions=[_slab[terminal_idx].position+shift]))
        if side!="up":
            water.rotate(180,"x")
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=shift[2], adsorbate=water, lateral_shift=shift[:2])
            
        return _slab


class HaliteSlab(Slab):
    
    
    #TODO
    #e_metal = None
    
    @property
    def e_metal(self):
        
        eles = list(set(self.symbols))
        eles.remove("O")
        try:
            eles.remove("H")
        except:
            pass
        e_metal = eles.pop()
        return e_metal
    
    @classmethod
    def get_slab(cls, primitive, indices: tuple, n_layers, lateral_repeat: tuple=(3, 4), vacuum=10.0):
        h, k, l = indices
        entry = str(h)+str(k)+str(l)
        method_entry = {
            "100":cls.halite_slab_100,
            "110": cls.halite_slab_110,
            #"111":self.halite_slab_111,
        }
        
        method = method_entry.get(entry, None)
        if method is None:
            raise ValueError("Current Miller Index has not implemented yet")

        slab = method(primitive=primitive, n_layers=n_layers, lateral_repeat=lateral_repeat, vacuum=vacuum)

        return slab
    
    @classmethod
    def halite_slab_100(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
        _surface = surface(primitive, (1, 0, 0), n_layers, vacuum)
        _surface = cls(_surface)
        for i in range(n_layers):
            _surface = _surface.del_surf_layer(tolerance=0.1, dsur='up')
        
        if vacuum is not None:
            _surface.center(vacuum=vacuum, axis=2)
        
        _slab = _surface * (lateral_repeat[0], lateral_repeat[1], 1)
        _slab.indices = "100"
        _slab.primitive = primitive
        _slab.n_layers = n_layers
        _slab.lateral_repeat = lateral_repeat
        return _slab
    
    @classmethod
    def halite_slab_110(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
        _surface = surface(primitive, (1, 1, 0), n_layers, vacuum=0)
        _surface = cls(_surface)
        for i in range(n_layers):
            _surface = _surface.del_surf_layer(tolerance=0.1, dsur='up')
        if vacuum is not None:
            _surface.center(vacuum=vacuum, axis=2)
    
        _slab = _surface * (lateral_repeat[0], lateral_repeat[1], 1)
        
        #TODO: 以后整理到init
        _slab.indices = "110"
        _slab.primitive = primitive
        _slab.n_layers = n_layers
        _slab.lateral_repeat = lateral_repeat
        return _slab

    
#     def halite_slab_111(self, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
#         _surface = surface(self, (1, 1, 1), n_layers, vacuum=0)

#         for i in range(n_layers):
#             _surface = _surface.del_surf_layer(tolerance=0.1, dsur='up')
        
#         if vacuum is not None:
#             _surface.center(vacuum=vacuum, axis=2)
        
#         return _surface #* (lateral_repeat[0], lateral_repeat[1], 1)
    
    def divide_terminal_idxs(self, idxs):

        #slab will be create within vacuum
        #we can divde a series of idxs to  
        half_z = self.cell[2][2]/2
        
        upper_idxs = list(filter(lambda x : self[x].position[2] >= half_z, idxs))
        bottom_idxs = list(filter(lambda x : self[x].position[2] < half_z, idxs))
        
        return bottom_idxs, upper_idxs
    
    def rerange_terminal_idxs(self, terminal_idxs, reshape=None, tolerance=0.1):
        # r_num, c_num to reshape the terminals 
        slab = self.copy()
        #tricky: shrink the box to manage the boundary atoms 
        assert tolerance >= 0
        old_cell = slab.cell.cellpar()
        old_cell[0] = old_cell[0] - tolerance
        old_cell[1] = old_cell[1] - tolerance
        slab.set_cell(old_cell)
        slab.pbc = True
        slab.wrap()
        
        terminal_idxs = np.array(terminal_idxs)
        ter_positions = slab[terminal_idxs].positions
        
        reranged = terminal_idxs[np.lexsort(ter_positions.T[:1])] #x，y reorder
        # reordered by x ,y
        
        if reshape:
            r_num = reshape[0]
            c_num = reshape[1]
            reranged = np.reshape(reranged, (r_num, c_num))

        return reranged

    def get_halite_metal_termials(self, cutoff=2.6):
        
        e_metal = self.e_metal
        
        #search the coord 5 terminals 
        # how to manage other coord atoms
        return search_coord_number(self, e_metal, "O", cutoff)[4]
    
    
    def get_adsorb_sites(self, dsur):
        n_layers = self.n_layers+3
        tmp = self.get_slab(self.primitive, 
                            indices=self.indices, 
                            n_layers=n_layers,
                            lateral_repeat=self.lateral_repeat)
        tmp = tmp.del_surf_layer(element=self.e_metal,dsur="dw")
        tmp = tmp.del_surf_layer(element="O",dsur="dw")
        #tricky
        tmp.cell = self.cell
        tmp.center(axis=2)
        tmp = tmp.del_surf_layer(element=self.e_metal,dsur="up")
        tmp = tmp.del_surf_layer(element=self.e_metal,dsur="dw")

#         idxs = tmp.positions[tmp.find_surf_idx(element="O", dsur=dsur)]
        return tmp.positions[tmp.find_surf_idx(element="O", dsur=dsur)]

    
    def get_halite100_split_water_symm(self,rank_idx=4,mode="a"):
        
        tmp = self.copy()
        wat_up = self.get_halite100_split_water(dsur="up",rank_idx=rank_idx, mode=mode)[-3:]
        wat_dw = self.get_halite100_split_water(dsur="dw",rank_idx=rank_idx, mode=mode)[-3:]
        tmp.extend(wat_up)
        tmp.extend(wat_dw)
        return tmp
    
    def get_halite100_split_water(self, dsur="up", rank_idx=4, mode="a"):
        
        h2o_lie = molecule("H2O")
        h2o_lie.rotate(90, 'y')
        h2o_lie.rotate(45, 'z')
        h2o_lie.rotate(180, 'z')
        oh = h2o_lie[:2]
        
        if dsur =="up":
            multi=1
        else:
            oh.rotate(180, 'z')
            multi=-1
        
        if mode !="a":
            oh.rotate(180,'z')
        
        
        pos = self.get_adsorb_sites(dsur=dsur)
        pos = pos[np.lexsort(pos.T[:2])] #x，y reorder
        rank_idx = min(len(pos), rank_idx)
        _pos = pos[rank_idx]
        print(_pos)
        #根据位点找到最近的4个氧 即为情况
        o_idxs = np.where(self.symbols == "O")[0]
        dist = get_distances(_pos, self.positions[o_idxs], self.cell, self.pbc)[1][0]
        o4_idxs = o_idxs[np.argsort(dist)[:4]]
        #按照x 方向对向量排序
        vs = get_distances(_pos, self.positions[o4_idxs], self.cell, self.pbc)[0][0].tolist()
        _sort = [_v[0] for _v in vs]
        _sort = np.argsort(_sort)

        #目前只取两个idx
        o2_idxs = [o4_idxs[_sort[0]],o4_idxs[_sort[-1]]]
        
        o_idx = o2_idxs[0]
        if mode != "a":
            o2_idxs = o2_idxs[::-1]
        if dsur !="up":
            o2_idxs = o2_idxs[::-1]
        o_idx = o2_idxs[0]

        dummy = Atoms("X",positions=[_pos])
        _slab = self.copy()
        _slab.extend(dummy)
        dummy_idx = len(_slab)-1
        _slab = _slab.add_adsorbate(dummy_idx, vertical_dist=0, adsorbate=oh)
        _slab = _slab.add_adsorbate(o_idx, vertical_dist=1*multi, adsorbate=Atoms("H"))
        del _slab[dummy_idx]
        return _slab

    def get_halite110_split_water_symm(self,rank_idx=4,mode="a"):
        
        tmp = self.copy()
        wat_up = self.get_halite110_split_water(dsur="up",rank_idx=rank_idx, mode=mode)[-3:]
        wat_dw = self.get_halite110_split_water(dsur="dw",rank_idx=rank_idx, mode=mode)[-3:]
        tmp.extend(wat_up)
        tmp.extend(wat_dw)
        return tmp
    
    def get_halite110_split_water(self, dsur="up", rank_idx=4, mode="a"):

        h2o_lie = molecule("H2O")
        h2o_lie.rotate(90, 'y')
        h2o_lie.rotate(45, 'z')
        h2o_lie.rotate(180, 'z')
        oh = h2o_lie[:2]
        
        if dsur =="up":
            multi=1
        else:
            oh.rotate(180, 'z')
            multi=-1
        
        if mode !="a":
            oh.rotate(180,'z')
        
        pos = self.get_adsorb_sites(dsur=dsur)
        pos = pos[np.lexsort(pos.T[:2])] #x，y reorder
        rank_idx = min(len(pos), rank_idx)
        _pos = pos[rank_idx]
        print(_pos)
        #根据位点找到最近的4个氧 即为情况
        o_idxs = np.where(self.symbols == "O")[0]
        dist = get_distances(_pos, self.positions[o_idxs], self.cell, self.pbc)[1][0]
        o4_idxs = o_idxs[np.argsort(dist)[:4]]
        #按照x 方向对向量排序
        vs = get_distances(_pos, self.positions[o4_idxs], self.cell, self.pbc)[0][0].tolist()
        _sort = [sum([_v[0], _v[1]]) for _v in vs]
        _sort = np.argsort(_sort)
        #目前只取两个idx
        o2_idxs = [o4_idxs[_sort[0]],o4_idxs[_sort[-1]]]
        
        
        if mode != "a":
            o2_idxs = o2_idxs[::-1]
        if dsur !="up":
            o2_idxs = o2_idxs[::-1]
        o_idx = o2_idxs[0]

        dummy = Atoms("X",positions=[_pos])
        _slab = self.copy()
        _slab.extend(dummy)
        dummy_idx = len(_slab)-1
        _slab = _slab.add_adsorbate(dummy_idx, vertical_dist=0, adsorbate=oh)
        _slab = _slab.add_adsorbate(o_idx, vertical_dist=1*multi, adsorbate=Atoms("H"))
        del _slab[dummy_idx]
        
        
        return _slab