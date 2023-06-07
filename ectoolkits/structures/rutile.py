import ase.io as io
from ase import Atoms
import numpy as np

from ectoolkits.structures.slab import Slab
from ase.build import surface, molecule
from typing import Tuple, List
from ase.geometry import get_distances

from .utils import search_coord_number, get_uniform_idxs

class Rutile(Slab):
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

    negative = None
    positive = None
    
    def copy(self):
        
        _slab = Slab.copy(self)
        _slab.negative = self.negative
        _slab.positive = self.positive
    
        return _slab
    
    def _reset_rutile_attri(self, negative=None, positive=None):
        
        if negative:
            self.negative = negative
        if positive:
            self.positive = positive


    @classmethod
    def _slab100(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 3), vacuum=10.0):
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
            slab = slab.del_surf_layer(tolerance=0.1, dsur='up')
        slab = slab * (lateral_repeat[0], lateral_repeat[1], 1)

        # sort according the z value
        slab = slab[slab.positions.T[2].argsort()]

        slab._reset_slab_attri(
            primitive=primitive,
            indices="100",
            n_layers=n_layers,
            lateral_repeat=lateral_repeat,
        )

        return slab

    def _slab100_water_split(self, rank_idx, dsur="up", mode="a", tolerance=0.1):
        
        slab = self
        bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        rank_idx = min(rank_idx, len(bot_idxs))
        if dsur=="up":
            terminal_idxs = up_idxs
        else:
            terminal_idxs = bot_idxs
        terminal_idx = terminal_idxs[rank_idx]
        
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
        O_brs = slab.find_rutile_metal_terminal_Obr(terminal_idx)
        if dsur =="dw":
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
        if dsur == "up":
            multi=1
        else:
            multi=-1
            oh.rotate(180,"z")
        #print(multi) 
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=shift[2], adsorbate=oh, lateral_shift=shift[:2])
        _slab = _slab.add_adsorbate(O_br, vertical_dist=1*multi, adsorbate=Atoms("H"),)
        
        return _slab[-3:]

    def _slab100_water_on_surface(self, terminal_idx, dsur, mode="a"):

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
            shift -= get_distances(slab.positions[terminal_idx], 
                                   slab.positions[idx], 
                                   cell=slab.cell, 
                                   pbc=slab.pbc)[0][0][0]    
        _slab = slab.copy()
        #_slab.extend(Atoms("N",positions=[_slab[terminal_idx].position+shift]))
        if dsur!="up":
            water.rotate(180,"x")
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=shift[2], adsorbate=water, lateral_shift=shift[:2])
            
        return _slab[-3:]



    @classmethod
    def _slab110(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
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
        slab._reset_slab_attri(
            primitive=primitive,
            indices="110",
            n_layers=n_layers,
            lateral_repeat=lateral_repeat,
        )

        return slab

    def _slab110_water_split(self, rank_idx=4, dsur="up", mode="fake"):
        
        #TODO: 临时这样实现 
        # 暂时没有分成两种情况
        if mode == "fake":
            print("current slab110 split water method has not distinguished to 2 modes. ")
        
        from ecflow_bandalign.tools.water import get_adsorb_rutile_struct_water

        e_metal = self.positive
        waters = get_adsorb_rutile_struct_water(self, e_metal, oxide_element="O", chosen_site=rank_idx)
        if dsur == "up":
            wat = waters[:3]
        elif dsur == "dw":
            wat = waters[3:]
            
        return wat
    
    def _slab110_water_on_surface(self, terminal_idx, dsur, **kwargs):
        
        #暂不实现mode=a,b
        slab = self.copy()
        #assert mode in ("a","b")
        #build a tempalte h2o
        water = molecule("H2O")
        water.rotate(90,"x")
        water.rotate(90,"y")
        water.rotate(38,"z")
        
        
        #oh = water
        
        #找到对称的位点 O_far 取负shift
        O_idxs = np.where(slab.symbols=="O")[0]
        dists =slab.get_distances(terminal_idx, O_idxs, mic=True)
        O_idxs_ard = O_idxs[np.argsort(dists)][:5]
        shift = np.array([0.,0.,0.])
        for idx in O_idxs_ard:
            shift -= get_distances(slab.positions[terminal_idx], 
                                   slab.positions[idx], 
                                   cell=slab.cell, 
                                   pbc=slab.pbc)[0][0][0]    
        _slab = slab.copy()
        #_slab.extend(Atoms("N",positions=[_slab[terminal_idx].position+shift]))
        if dsur!="up":
            water.rotate(180,"x")
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=shift[2], adsorbate=water, lateral_shift=shift[:2])
        
        return _slab[-3:]

    def _slab110_split_water_in_converage(self, coverage_percent=0.5, ):
        
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



    @classmethod
    def _slab001(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 2), vacuum=10.0):
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
        
        slab._reset_slab_attri(
            primitive=primitive,
            indices="001",
            n_layers=n_layers,
            lateral_repeat=lateral_repeat,
        )

        return slab
    
    #TODO no water method imple yet


    @classmethod
    def _slab101(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 2), vacuum=10.0):
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
        slab._reset_slab_attri(
            primitive=primitive,
            indices="101",
            n_layers=n_layers,
            lateral_repeat=lateral_repeat,
        )
        
        return slab

    
    def _slab101_water_split(self, rank_idx, dsur, mode="a", tolerance=0.1):
        
        
        #根据顺序位点 选择要加水的地方
        bot_idxs, up_idxs = self.get_reranged_terminals_two_sides(tolerance=tolerance)
        rank_idx = min(rank_idx, len(bot_idxs))
        if dsur=="up":
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
        
        if dsur == "up":
            multi=1
        elif dsur == "dw":
            multi=-1
            O_brs = O_brs[::-1]
            oh.rotate(180,"z")
        O_br = O_brs[0] #choose the O_br to ad H atom
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=2*multi , adsorbate=oh)
        _slab = _slab.add_adsorbate(O_br, vertical_dist=1*multi, adsorbate=Atoms("H"))
        
        return _slab[-3:]
    
    def _slab101_water_on_surface(self, terminal_idx, dsur, mode="a", height=2.2):
        
        #TODO: 与sphalite的输入参数形式不一致， 这里需要的是idx 而不是位置
        # 原因在这里根据金属位点 寻找的O位点， 而不是直接知道O的位置
        
        _slab = self.copy()
        
        water = molecule("H2O")
        water.rotate(90,"x")
        water.rotate(90,"y")
        water.rotate(170,"z")
        if mode == "b":
            water.rotate(205,"z")
        
        if dsur=="up":
            multi=1
        else:
            multi=-1
            water.rotate(180,"z")
        
        _slab = _slab.add_adsorbate(terminal_idx, vertical_dist=height*multi , adsorbate=water)
        
        return _slab[-3:]
    
    
    
    ############################# contruct method end   #######################
    
    @classmethod
    def get_slab(self, primitive, indices: Tuple[int], n_layers, lateral_repeat: Tuple[int]=(2, 4), vacuum=10.0):
        h, k, l = indices
        entry = str(h)+str(k)+str(l)
        method_entry = {
            "110": self._slab110,
            "001": self._slab001,
            "100": self._slab100,
            "101": self._slab101,
        }
        
        method = method_entry.get(entry, None)

        method = method_entry.get(entry, None)
        if method is None:
            raise ValueError("Current Miller Index has not implemented yet")

        # if method is None:
        #     raise ValueError("Current Miller Index has not implemented yet")
        slab = method(primitive=primitive, n_layers=n_layers, lateral_repeat=lateral_repeat, vacuum=vacuum)

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
        
        e_metal = self.positive
        
        #search the coord 5 terminals 
        # how to manage other coord atoms
        return search_coord_number(self, e_metal, self.negative, cutoff)[5]
    
    def find_rutile_metal_terminal_Obr(self, m_terminal_idx):
        
        #find the coord 5 terminal O_br idx
        slab = self

        e_metal = self.positive
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


    #除了工厂方法 和sphale是一致的
    #TODO: 以后用多态实现
    def get_water_split_symm(self, rank_idx=4, mode="a", **kwargs): #kwargs like tolerance?
        
        h, k, l = self.indices
        entry = str(h)+str(k)+str(l)
        entries = {
            "100":self._slab100_water_split,
            "110":self._slab110_water_split,
            "101":self._slab101_water_split,
        }

        _method = entries[entry]
        #find the up dw waters in the slab
        wat_up = _method(dsur="up", rank_idx=rank_idx, mode=mode, **kwargs)
        wat_dw = _method(dsur="dw", rank_idx=rank_idx, mode=mode, **kwargs)
        
        return (wat_up, wat_dw)

    def get_slab_water_split_symm(self, rank_idx=4, mode="a", **kwargs):
        
        tmp = self.copy()
        wats = self.get_water_split_symm(rank_idx=rank_idx, mode=mode, **kwargs)
        tmp.extend(wats[0])
        tmp.extend(wats[1])

        return tmp
    
    
    def _get_water_on_surface(self, terminal_idx, dsur="up", mode="a", **kwargs):
        
        h, k, l = self.indices
        entry = str(h)+str(k)+str(l)
        entries = {
            "100":self._slab100_water_on_surface,
            "110":self._slab110_water_on_surface,
            "101":self._slab101_water_on_surface,
        }
        _method = entries[entry]
        wat = _method(terminal_idx=terminal_idx, dsur=dsur, mode="a", **kwargs)
        
        return wat
        

    def get_water_on_surface_covered(self, mode="a", **kwargs):
        
        idxs_dw, idxs_up = self.get_reranged_terminals_two_sides()
        ranks = len(idxs_dw)
        
        waters = self[:0]
        for _idx in idxs_up:
            #print(_idx)
            _wat = self._get_water_on_surface(_idx, dsur="up", mode=mode, **kwargs)
            waters.extend(_wat)
        for _idx in idxs_dw:
            #print(_idx)
            _wat = self._get_water_on_surface(_idx, dsur="dw", mode=mode, **kwargs )
            waters.extend(_wat)
        
        return waters

    def get_slab_water_on_surface_covered(self, mode="a", **kwargs):
        
        tmp = self.copy()
        wats = self.get_water_on_surface_covered(mode=mode, **kwargs)
        
        return tmp + wats

