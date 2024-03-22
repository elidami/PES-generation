from pymatgen.core.sites import PeriodicSite
import numpy as np
from pymatgen.core.surface import center_slab, Structure

def flip_slab(slab):
    """
    Flip the z coordinates of the input slab by multiplying all z-coords with -1.

    Parameters
    ----------
    slab : pymatgen.core.surface.Slab
       The input slab object flip

    Returns
    -------
    flipped_slab : pymatgen.core.surface.Slab
        The flipped slab

    """
    flip_matrix = np.array([[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., -1.]])
    flipped_coords = np.dot(slab.cart_coords, flip_matrix)
        
    flipped_slab = Structure(lattice=slab.lattice,
                        species=slab.species,
                        coords=flipped_coords,
                        #miller_index=slab.miller_index,
                        #oriented_unit_cell=slab.oriented_unit_cell,
                        #shift=slab.shift,
                        #scale_factor=slab.scale_factor,
                        #reconstruction=slab.reconstruction,
                        coords_are_cartesian=True,
                        site_properties=slab.site_properties)
    return center_slab(flipped_slab)
    

def clean_up_site_properties(structure):
    """
    Cleans up site_properties of structures that contain NoneTypes.
    
    If an interface is created from two different structures, it is possible
    that some site properties like magmom are not set for both structures.
    This can lead later to problems since they are replaced by None.
    This function replaces NoneTypes with 0.0 for magmom and deletes all other
    site_properties if None entries are found in it.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Input structure

    Returns
    -------
    struct : pymatgen.core.structure.Structure
        Output structure

    """
    struct = structure.copy()
    for key in struct.site_properties.keys():
        if key == 'magmom':
            new_magmom = []
            for m in struct.site_properties[key]:
                if m == None:
                    new_magmom.append(0.0)
                else:
                    new_magmom.append(m)
            struct.add_site_property('magmom', new_magmom)
        else:
            if any(struct.site_properties[key]) is None:
                struct.remove_site_property(key)
    return struct

def stack_aligned_slabs(bottom_slab, top_slab, top_shift=[0,0,0]):
    """
    Combine slabs that are centered around 0 into a single structure.
    
    Optionally shift the top slab by a vector of cartesian coordinates.

    Parameters
    ----------
    bottom_slab : pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
        Bottom slab.
    top_slab : pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
        Top slab.
    top_shift : list of 3 floats, optional
        Vector of caresian coordinates with which to shift the top slab.
        The default is [0,0,0].

    Returns
    -------
    interface : pymatgen.core.structure.Structure or pymatgen.core.surface.Slab
                depending on type of bottom_slab
        An interface structure of two slabs with an optional shift of the top
        slab.

    """
    interface = bottom_slab.copy()
    t_copy = top_slab.copy()
    
    t_copy.translate_sites(indices=range(len(t_copy.sites)),
                           vector=top_shift,
                           frac_coords=False, to_unit_cell=False)
    
    for s in t_copy.sites: #per ogni sito nel top_slab traslato...
        new_site = PeriodicSite(lattice=interface.lattice,
                                coords=s.frac_coords,
                                coords_are_cartesian=False,
                                species=s.species,
                                properties=s.properties)
        interface.sites.append(new_site)
    
    return interface

def recenter_aligned_slabs(top_slab, bot_slab, d=2.5):
    """
    Center two slabs around z=0 and give them the distance d if provided.

    Parameters
    ----------
    top_slab : pymatgen.core.structure.Structure
        The slab that should be on top.

    bot_slab : pymatgen.core.structure.Structure
        The slab that should be on the bottom.

    d : float, optional
        The desired distance between the slabs. The default is 2.5.

    Returns
    -------
    t_copy : pymatgen.core.structure.Structure
        Top slab that is shifted so that the lowest atom is at +d/2
    b_copy : pymatgen.core.structure.Structure
        Bottom slab that is shifted so that the topmost atom is at -d/2

    """
    t_copy = top_slab.copy()
    b_copy = bot_slab.copy()
    top_zs=[]
    bot_zs=[]
    for s in t_copy.sites:
        top_zs.append(s.coords[-1])
    top_shift = -min(top_zs) + d/2
    
    for s in b_copy.sites:
        bot_zs.append(s.coords[-1])
    bot_shift = -max(bot_zs) - d/2

    t_copy.translate_sites(indices=range(len(t_copy.sites)),
                           vector=[0, 0, top_shift],
                           frac_coords=False, to_unit_cell=False)
    b_copy.translate_sites(indices=range(len(b_copy.sites)),
                           vector=[0, 0, bot_shift],
                           frac_coords=False, to_unit_cell=False)
    return t_copy, b_copy