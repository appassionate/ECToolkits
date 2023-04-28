import ase.io as io
from ase import Atoms

from ectoolkits.structures.slab import Slab
from ase.build import surface, molecule
from typing import Tuple, List


class Sphalerite(Slab):
    
    # neg and pos should be single
    # which means it cant deal with complex 
    negative = None
    positive = None
    
    def copy(self):
        
        spha = Slab.copy(self)
        spha.negative = self.negative
        spha.positive = self.positive
    
        return spha
    
    @property
    def get_slab_method(self):
        
        return {
            "100":None,
            "110":self._get_slab110,
        }
    
    @classmethod
    def _slab110(cls, primitive, n_layers=5, lateral_repeat: tuple=(2, 4), vacuum=10.0):
        
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

    def _slab110_surface_water(self, position, side, mode, height=0):
        
        _slab = self.copy()
        
        water = molecule("H2O")
        water.rotate(90,"x")
        water.rotate(90,"y")
        water.rotate(180,"z")
        if mode == "b":
            water.rotate(180,"z")
        
        if side=="up":
            multi=1
        else:
            multi=-1
            water.rotate(180,"z")
        _pos = position
        dummy = Atoms("X",positions=[_pos])
        _slab = self.copy()
        _slab.extend(dummy)
        dummy_idx = len(_slab)-1
        _slab = _slab.add_adsorbate(dummy_idx, vertical_dist=multi*height, adsorbate=water)
        del _slab[dummy_idx]
        
        return _slab[-3:]
    
    
    
    @classmethod
    def get_slab(cls, primitive, indices: Tuple[int], n_layers:int, lateral_repeat=(2,2), vacuum=10.0):
        h, k, l = indices
        entry = str(h)+str(k)+str(l)
        method_entry = {
            "100":None,
            "110": cls._slab110,
        }
        
        method = method_entry.get(entry, None)
        if method is None:
            raise ValueError("Current Miller Index has not implemented yet")

        slab = method(primitive=primitive, n_layers=n_layers, lateral_repeat=lateral_repeat, vacuum=vacuum)

        return slab
    
    def get_adsorb_sites(self, dsur):
        n_layers = self.n_layers+3
        tmp = self.get_slab(self.primitive, 
                            indices=self.indices, 
                            n_layers=n_layers,
                            lateral_repeat=self.lateral_repeat)
        tmp = tmp.del_surf_layer(element=self.positive,dsur="dw")
        tmp = tmp.del_surf_layer(element=self.negative,dsur="dw")
        #tricky
        tmp.cell = self.cell
        tmp.center(axis=2)
        tmp = tmp.del_surf_layer(element=self.positive,dsur="up")
        tmp = tmp.del_surf_layer(element=self.positive,dsur="dw")

        idxs = tmp.find_surf_idx(element=self.negative, dsur=dsur)
        pos = tmp.positions[tmp.find_surf_idx(element=self.negative, dsur=dsur)]
        return (pos, idxs)
        #return tmp
    
    def _get_surface_water(self, position, side, mode, height=0):
        
        #TODO
        #more interface imple need 
        water = self._slab110_surface_water(position=position, 
                                            side=side, mode=mode, 
                                            height=height)
        
        return water
    
    def get_surface_water_covered(self, height=0, mode="a"):
        
        pos_dw = self.get_adsorb_sites("dw")[0] #just use pos
        pos_up = self.get_adsorb_sites("up")[0]
        
        waters = self[:0]
        for _pos in pos_up:
            print(_pos)
            _wat = self._get_surface_water(_pos, side="up", mode=mode, height=height)
            waters.extend(_wat)
        for _pos in pos_dw:
            _wat = self._get_surface_water(_pos, side="dw", mode=mode, height=height)
            waters.extend(_wat)
        
        return waters