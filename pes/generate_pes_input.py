from uuid import uuid4
import numpy as np
from manipulate_struct import recenter_aligned_slabs, stack_aligned_slabs, clean_up_site_properties

def generate_pes_inputs(bot_slab, top_slab, shifts):
    """
    Prepare the inputs to run VASP on the fly for the different relative
    lateral positions that are asked for the interface.

    """

    structures = []
    # Loop over all the shifts in order to do them one by one
    for s in shifts.keys():

        # Build the interface by applying a single shifts
        inter = apply_interface_shifts(bot_slab, top_slab, shifts[s])
        
        # Store the data for each simulation
        structures.append(inter)
    
    return structures

def apply_interface_shifts(bot_slab, top_slab, shifts):
    """
    Create a list of interfaces, by combining a bot_slab (substrate) to a 
    top_slab (coating) and placing them in different relative lateral
    positions by applying different lateral shift to the upper slab. 

    Parameters
    ----------
    bot_slab : pymatgen.core.surface.Slab
        Bottom slab structure (substrate).

    top_slab : pymatgen.core.surface.Slab
        Top slab structure (coating).

    shifts : np.ndarray or list of lists
        Lateral shifts that needs to be applied to the upper slab on top of the
        lower one, in order to achieve all the relative lateral configurations
        between the two structures. It can be either a list of two-elements
        lists or a numpy matrix of shape nx2.
        The structure should resemble: [[x0, y0], [x1, y1], ..., [xn, yn]].

    Returns
    -------
    interfaces: (list of) pymatgen.core.surface.Slab
        All the interfaces created matching the slabs according to shifts.

    """

    # Convert shifts to a numpy array
    shifts = np.array(shifts)
    
    # Recenter the slabs to the center of the cell
    top_slab, bot_slab = recenter_aligned_slabs(top_slab, bot_slab)
    
    # Create and save the different interfaces
    interfaces = []
    for s in shifts:
        inter_struct = stack_aligned_slabs(bot_slab, top_slab, [s[0], s[1], 0])
        interfaces.append(clean_up_site_properties(inter_struct))
    
    # Just return the interface object if only a shift is required
    if shifts.shape[0] == 1:
        interfaces = interfaces[0]
    
    return interfaces