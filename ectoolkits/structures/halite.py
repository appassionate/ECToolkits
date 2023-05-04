import ase.io as io
from ase import Atoms
import numpy as np

from ectoolkits.structures.slab import Slab
from ase.build import surface, molecule
from typing import Tuple, List
from ase.geometry import get_distances


class Halite(Slab):
    
    negative = None
    positive = None
    
    
    def copy(self):
        
        _slab = Slab.copy(self)
        _slab.negative = self.negative
        _slab.positive = self.positive
    
        return _slab
    
    def _reset_halite_attri(self, negative=None, positive=None):
        
        if negative:
            self.negative = negative
        if positive:
            self.positive = positive
    
    @classmethod
    def _slab_100(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
        _surface = surface(primitive, (1, 0, 0), n_layers, vacuum)
        _surface = cls(_surface)
        for i in range(n_layers):
            _surface = _surface.del_surf_layer(tolerance=0.1, dsur='up')
        
        if vacuum is not None:
            _surface.center(vacuum=vacuum, axis=2)
        
        _slab = _surface * (lateral_repeat[0], lateral_repeat[1], 1)
        _slab._reset_slab_attri(
            primitive=primitive,
            indices="100",
            n_layers=n_layers,
            lateral_repeat=lateral_repeat,
            
        )
        return _slab
    
    def _slab100_water_split(self, dsur="up", rank_idx=4, mode="a"):
        
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
        
        # rank_idx 在这个底层方法里有必要出现吗？
        pos = self.get_adsorb_sites(dsur=dsur)
        
        #TODO: 改成pos?
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
        
        return _slab[-3:]

    def _slab100_water_on_surface(self):
        return NotImplementedError()    
    
    
    @classmethod
    def _slab_110(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
        _surface = surface(primitive, (1, 1, 0), n_layers, vacuum=0)
        _surface = cls(_surface)
        for i in range(n_layers):
            _surface = _surface.del_surf_layer(tolerance=0.1, dsur='up')
        if vacuum is not None:
            _surface.center(vacuum=vacuum, axis=2)
    
        _slab = _surface * (lateral_repeat[0], lateral_repeat[1], 1)
        
        _slab._reset_slab_attri(
            primitive=primitive,
            indices="110",
            n_layers=n_layers,
            lateral_repeat=lateral_repeat,
            
        )
        
        return _slab
    
    def _slab110_water_split(self, dsur="up", rank_idx=4, mode="a"):

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
        
        
        return _slab[-3:]
    
    def _slab110_water_on_surface(self):
        return NotImplementedError()    
    
#     def halite_slab_111(self, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
#         _surface = surface(self, (1, 1, 1), n_layers, vacuum=0)

#         for i in range(n_layers):
#             _surface = _surface.del_surf_layer(tolerance=0.1, dsur='up')
        
#         if vacuum is not None:
#             _surface.center(vacuum=vacuum, axis=2)
        
#         return _surface #* (lateral_repeat[0], lateral_repeat[1], 1)
    

    @classmethod
    def get_slab(cls, primitive, indices: tuple, n_layers, lateral_repeat: tuple=(3, 4), vacuum=10.0):
        h, k, l = indices
        entry = str(h)+str(k)+str(l)
        method_entry = {
            "100":cls._slab_100,
            "110": cls._slab_110,
            #"111":self.halite_slab_111,
        }
        
        method = method_entry.get(entry, None)
        if method is None:
            raise ValueError("Current Miller Index has not implemented yet")

        slab = method(primitive=primitive, n_layers=n_layers, lateral_repeat=lateral_repeat, vacuum=vacuum)

        return slab
    
    def divide_terminal_idxs(self, idxs):

        #slab will be create within vacuum
        #we can divde a series of idxs to  
        half_z = self.cell[2][2]/2
        
        upper_idxs = list(filter(lambda x : self[x].position[2] >= half_z, idxs))
        bottom_idxs = list(filter(lambda x : self[x].position[2] < half_z, idxs))
        
        return bottom_idxs, upper_idxs
    
    # def rerange_terminal_idxs(self, terminal_idxs, reshape=None, tolerance=0.1):
    #     # r_num, c_num to reshape the terminals 
    #     slab = self.copy()
    #     #tricky: shrink the box to manage the boundary atoms 
    #     assert tolerance >= 0
    #     old_cell = slab.cell.cellpar()
    #     old_cell[0] = old_cell[0] - tolerance
    #     old_cell[1] = old_cell[1] - tolerance
    #     slab.set_cell(old_cell)
    #     slab.pbc = True
    #     slab.wrap()
        
    #     terminal_idxs = np.array(terminal_idxs)
    #     ter_positions = slab[terminal_idxs].positions
        
    #     reranged = terminal_idxs[np.lexsort(ter_positions.T[:1])] #x，y reorder
    #     # reordered by x ,y
        
    #     if reshape:
    #         r_num = reshape[0]
    #         c_num = reshape[1]
    #         reranged = np.reshape(reranged, (r_num, c_num))

    #     return reranged

    # def get_halite_metal_termials(self, cutoff=2.6):
        
    #     e_metal = self.e_metal
        
    #     #search the coord 5 terminals 
    #     # how to manage other coord atoms
    #     return search_coord_number(self, e_metal, "O", cutoff)[4]
    
    
    def get_adsorb_sites(self, dsur):
        n_layers = self.n_layers+3
        tmp = self.get_slab(self.primitive, 
                            indices=self.indices, 
                            n_layers=n_layers,
                            lateral_repeat=self.lateral_repeat)
        tmp = tmp.del_surf_layer(element=self.positive, dsur="dw")
        tmp = tmp.del_surf_layer(element="O",dsur="dw")
        #tricky
        tmp.cell = self.cell
        tmp.center(axis=2)
        tmp = tmp.del_surf_layer(element=self.positive,dsur="up")
        tmp = tmp.del_surf_layer(element=self.positive,dsur="dw")

#         idxs = tmp.positions[tmp.find_surf_idx(element="O", dsur=dsur)]
        return tmp.positions[tmp.find_surf_idx(element="O", dsur=dsur)]

    #TODO: duplicate it
    def get_split_water_symm(self, rank_idx=4, mode="a", **kwargs):
        return self.get_water_split_symm(self, rank_idx=rank_idx, mode=mode, **kwargs)
    
    def get_water_split_symm(self, rank_idx=4, mode="a", **kwargs):
        
        h, k, l = self.indices
        entry = str(h)+str(k)+str(l)
        entries = {
            "100":self._slab100_water_split,
            "110":self._slab110_water_split,
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
    
    def _get_water_on_surface(self):
        pass
    def get_water_on_surface_covered(self):
        return NotImplementedError()
    def get_slab_water_on_surface_covered(self):
        return NotImplementedError()