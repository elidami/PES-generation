import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.structure_matcher import StructureMatcher
from ase import Atoms
from manipulate_struct import stack_aligned_slabs, \
    clean_up_site_properties, recenter_aligned_slabs

# =============================================================================
# CALCULATE THE HS POINTS FOR A SLAB
# =============================================================================

def get_slab_hs(slab, allowed_sites=['ontop', 'bridge', 'hollow'], to_array=False):
    adsf = AdsorbateSiteFinder(slab, height= 0.3)

    # Extract the unique HS points for the given surface in the unit cell
    unique_sites = adsf.find_adsorption_sites( distance=0,
                                               symm_reduce=0.01,
                                               near_reduce=0.01,
                                               no_obtuse_hollow=True )
    #print("unique_sites: ", unique_sites)
    #Extract all the HS points for the given surface in the lattice cell
    replica_sites = adsf.find_adsorption_sites( distance=0,
                                                symm_reduce=0,
                                                near_reduce=0.01,
                                                no_obtuse_hollow=True )
    
    #print("replica_sites: ", replica_sites) #1 bridge in più
    # Identify the unique HS points of the slab and rename them
    hs = {}
    for key in allowed_sites:
        if unique_sites[key] != []:
            for n, data in enumerate(unique_sites[key]):
                hs[key+'_'+str(n+1)] = data
   
    # Recognize of which unique HS point each site is the replica
    hs_all = {}
    for k in hs.keys():
        hs_all[k] = []
    #print(hs_all.keys())
    for key in hs_all.keys():
        n=0
        for site in replica_sites['all']:
            n=n+1
            pts_to_evaluate = [hs[key].copy()]
            pts_to_evaluate.append(np.array(site))
            #pts = adsf.symm_reduce(pts_to_evaluate)
            pts = adsf.symm_reduce(pts_to_evaluate, threshold=1e-2) #Reduces the set of adsorbate sites by finding removing symmetrically equivalent duplicates.
            #print("key: ", key)
            #print("site: ", site)
            #print(n,len(pts))
            if len(pts) == 1:
                hs_all[key].append(site) 
                                         
    hs = normalize_hs_dict(hs, to_array)
    hs_all = normalize_hs_dict(hs_all, to_array)

    return hs, hs_all

def normalize_hs_dict(hs, to_array=True):
    """
    Convert the hs elements returned by get_slab_hs to proper np.array or lists
    Important to use with unfolded HS points, which are lists of arrays
    
    """
    
    hs_new = {}
    
    for k in hs.keys():
        data = hs[k]
        
        # Default: convert everything in a list
        if isinstance(data, list):
            elements_list = []
            for element in data:
                elements_list.append(list(element))
            hs_new[k] = elements_list
        else:
            hs_new[k] = [list(data)]
        
        # Convert to array, default and smart choice
        if to_array == True:
            hs_new[k] = np.array(hs_new[k])
    
    return hs_new

def hs_dict_converter(hs, to_array=True):
    """
    Modify the type of the elements of the HS dictionary to list or np.ndarray.

    Parameters
    ----------
    hs : dict
        Dictionary containing the High Symmetry points.
    to_array : bool, optional
        If set to True convert to array, otherwise convert to list. 
        The default is True.

    Raises
    ------
    ValueError
        Raised if the dictionary values are of different types.
        Print to stdout: "Your dictionary is weird, values have mixed types"

    Returns
    -------
    hs_new : dict
        New HS dictionary converted to the desired type.

    """
    
    hs_new = {}
    dict_types = list( set(type(k) for k in hs.values()) ) #crea una lista che contiene tutti i tipi unici dei valori di hs

    try: 
        assert(len(dict_types) == 1)
        
        typ = dict_types[0]
        if to_array:
            if typ == list:
                for k in hs.keys():
                    hs_new[k] = np.array(hs[k])    
            else:
                return hs
            
        else:
            if typ == np.ndarray:
                for k in hs.keys():
                    hs_new[k] = hs[k].tolist() 
            else:
                return hs
            
        return hs_new
            
    except:
        raise ValueError('Your dictionary is weird, values have mixed types')


# ==========================================================================
# CALCULATE HS POINTS FOR AN INTERFACE
# ==========================================================================

def get_interface_hs(hs_1, hs_2, cell, to_array=False, z_red=True):
    """
    Calculate the HS sites for a hetero interface by combining the HS sites of
    the bottom slab (hs_1) with the upper slab (hs_2) 

    Parameters
    ----------
    hs_1 : dict
        High Symmetry points of the bottom slab
    hs_2 : dict
        High Symmetry points of the upper slab
    to_array : bool, optional
        If set to True return an HS dictionary containing arrays, else lists
        The default is False.
    z_red : bool, optional
        Remove the z-coordinates from the translations. The default is True.

    Returns
    -------
    hs : dict
        High Symmetry points of the hetero interface.

    """
    
    hs = {}
    
    typ_1 = list( set(type(k) for k in hs_1.values()) )[0]
    if typ_1 == list:
        hs_1 = hs_dict_converter(hs_1, to_array=True)
        
    typ_2 = list( set(type(k) for k in hs_1.values()) )[0]
    if typ_2 == list:
        hs_2 = hs_dict_converter(hs_2, to_array=True)
        
    
    # Calculate the shift between each HS point of the first material with each
    # HS point of the second material
    for k1, v1 in hs_1.items():
        for k2, v2 in hs_2.items():  
            shifts_stack = []
            for el_d1 in v1: #questo ciclo qua serve per essere generale e far sì che funzioni anche quando consideriamo hs_all
                shifts_stack.append( v2 - el_d1 )
                
            hs[k1+'-'+k2] = np.concatenate(shifts_stack, axis=0)
    
    hs = pbc_hs_points(hs, cell, to_array=to_array, z_red=z_red) 
    return hs 

def pbc_hs_points(hs, cell, to_array=False, z_red=True):
    """
    Create a "fake" molecule structure from the HS points calculated for
    the tribological interface, in order to apply PBC to the sites.
    Return a dictionary with the sites within cell. z_red remove the 
    z-coordinates from the translations.

    """
    
    # Type check and error handling
    if not isinstance(hs, dict):
            raise TypeError("hs must be a dictionary")
    
    # Convert the hs elements to lists if necessary
    typ = list( set(type(k) for k in hs.values()) )
    if len(typ) > 1:
        raise ValueError('Your dictionary is weird, values have mixed types')
    elif typ[0] != list:
        hs = hs_dict_converter(hs, to_array=False)
    
    # Run over dictionary values and apply PBC
    hs_new = {}
    for k in hs.keys():
        sites = hs[k]
        
        # Create a fake atomic structures and apply PBC
        atoms_fake = Atoms( positions=sites, cell=cell, pbc=[1,1,1] )
        hs_new[k] = atoms_fake.get_positions( wrap=True, pbc=True )
        
        # Remove z coordinates
        if z_red:
            hs_new[k] = hs_new[k][:, :2]
            
    hs_new = hs_dict_converter(hs_new, to_array=to_array)
    
    return hs_new    


# ==========================================================================
# TOOLS FOR HS DICTIONARIES
# ==========================================================================

def fix_hs_dicts(hs_unique, hs_all, top_aligned, bot_aligned, ltol=0.01, 
                 stol=0.01, angle_tol=0.01, primitive_cell=False, scale=False):
    """
    Remove duplicate shifts from the hs points and assign the replicas correctly.
    
    A StructureMatcher is defined with the selected tolerances and options and
    then used to remove equivalent shifts from the high-symmetry points
    dictionaries and assign the replicated points correctly to their unique
    counterparts using the <remove_equivalent_shifts> and <assign_replicate_points>
    functions.

    Parameters
    ----------
    hs_unique : dict
        Unique high-symmetry points of the interface from <get_interface_hs>
    hs_all : dict
        Replicated high-symmetry points of the interface from <get_interface_hs>.
    top_aligned : pymatgen.core.surface.Slab or pymatgen.core.structure.Structure
        The top slab of the interface
    bot_aligned : pymatgen.core.surface.Slab or pymatgen.core.structure.Structure
        The bottom slab of the interfaces
    ltol : float, optional
       Fractional length tolerance. The default is 0.01.
    stol : float, optional
        Site tolerance. The default is 0.01.
    angle_tol : float, optional
        Angle tolerance in degrees. The default is 0.01.
    primitive_cell : bool, optional
        If true: input structures will be reduced to primitive cells prior to
        matching. The default is False.
    scale : bool, optional
        Input structures are scaled to equivalent volume if true; For exact
        matching, set to False. The default is False.

    Returns
    -------
    c_u : TYPE
        DESCRIPTION.
    c_all : TYPE
        DESCRIPTION.

    """
    top_slab, bot_slab = recenter_aligned_slabs(top_aligned,
                                                bot_aligned,
                                                d=4.5) #to center the slabs 

    struct_match = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                                    primitive_cell=primitive_cell, scale=scale)

    # Use the structure matcher to find shifts leading to equivalent interfaces
    # and pop these entries out of the dictionaries.
    c_u, c_a = remove_equivalent_shifts(hs_u=hs_unique.copy(),
                                        hs_a=hs_all.copy(),
                                        top_slab=top_slab,
                                        bot_slab=bot_slab,
                                        structure_matcher=struct_match)

    c_all = assign_replicate_points(hs_u=c_u,
                                    hs_a=c_a,
                                    top_slab=top_slab,
                                    bot_slab=bot_slab,
                                    structure_matcher=struct_match)

    return c_u, c_all 

def assign_replicate_points(hs_u, hs_a, top_slab, bot_slab, structure_matcher):
    """Assign the replicated high-symmetry points to the correct unique ones.
    
    Although most of the high-symmetry points should be assigned to the correct
    lable, there is the occasional shift that is equivalent for two lables.
    This function imploys the StructureMatcher to match the replicated points
    to their unique counterparts, so the energy can later be transfered
    correctly.
    

    Parameters
    ----------
    hs_u : dict
        Unique high-symmetry points of the interface.
    hs_a : dict
        All high-symmetry points of the interface.
    top_slab : pymatgen.core.surface.Slab or pymatgen.core.structure.Structure
        The top slab of the interface
    bot_slab : pymatgen.core.surface.Slab or pymatgen.core.structure.Structure
        The bottom slab of the interfaces
    structure_matcher : pymatgen.analysis.structure_matcher.StructureMatcher
        Class to find equivalent structures (mirrors, rotations, etc...)

    Returns
    -------
    new_hsp_dict_a : dict
        All high Symmetry points of the interface without duplicated entries.

    """
    all_shifts = []
    for key, value in hs_a.items():
        if all_shifts == []:
            all_shifts = value
        else:
            all_shifts = np.concatenate([all_shifts, value], axis=0).tolist()
    all_shifts = np.unique(all_shifts, axis=0) #unique rimuove i duplicati

    
    new_hsp_dict_a = {}
    for key, value in hs_u.items():
        unique_struct = stack_aligned_slabs(bot_slab, top_slab, 
                                            [value[0][0], value[0][1], 0])
        unique_struct = clean_up_site_properties(unique_struct)
        
        for shift in all_shifts:
            test_struct = stack_aligned_slabs(bot_slab,
                                              top_slab,
                                              top_shift = [shift[0],
                                                           shift[1],
                                                           0])
            test_struct = clean_up_site_properties(test_struct)
            if structure_matcher.fit(unique_struct, test_struct):
                new_hsp_dict_a.setdefault(key, []).append(shift)
    
    return new_hsp_dict_a

def remove_equivalent_shifts(hs_u, hs_a, top_slab, bot_slab, structure_matcher): 
    """
    Remove equivalent shifts from an interface high-symmetry point dict.
    
    When the high-symmetry points of two slabs are combined by finding all t
    combinations (e.g. ontop_1-hollow_2, ontop_1-bridge_1, ...) by a 
    get_interface_hs a lot of duplicates might be created. Here we use a
    pymatgen.analysis.structure_matcher.StructureMatcher to get rid of these
    duplicates both in the unique and the replicated high symmetry points.

    Parameters
    ----------
    hs_u : dict
        Unique high-symmetry points of the interface.
    hs_a : dict
        All high-symmetry points of the interface.
    top_slab : pymatgen.core.surface.Slab or pymatgen.core.structure.Structure
        The top slab of the interface
    bot_slab : pymatgen.core.surface.Slab or pymatgen.core.structure.Structure
        The bottom slab of the interfaces
    structure_matcher : pymatgen.analysis.structure_matcher.StructureMatcher
        Class to find equivalent structures (mirrors, rotations, etc...)

    Returns
    -------
    hs_u : dict
        Unique high Symmetry points of the interface without equivalent entries.
    hs_a : dict
        All high Symmetry points of the interface without equivalent entries.

    """

    structure_list = {}
    for key, value in hs_u.items():
        x_shift = value[0][0] 
        y_shift = value[0][1] 
        inter_struct = stack_aligned_slabs(bot_slab,
                                           top_slab,
                                           top_shift = [x_shift, y_shift, 0]) #questo serve per creare l'interfaccia considerando la bot_slab e la top_slab shiftata in modo che gli HS siano allineati
        clean_struct = clean_up_site_properties(inter_struct) #mette =0 eventuali proprietà (i.e. magmom) che sono =None
        structure_list[key] = clean_struct

    equivalent_structs = {}
    doubles_found = []
    for name, struct in structure_list.items(): #name è e.g. bridge_1-ontop_1 mentre struct è la clean_struct: come se fosse un POSCAR
        #print(f"name: ", name)
        for name_2, struct_2 in structure_list.items():
            #print(f"name_2: ", name_2)
            if name != name_2:
                if structure_matcher.fit(struct, struct_2) and name not in doubles_found:
                    #print(f"boolean: ", structure_matcher.fit(struct, struct_2))
                    #print(f"doubles_found: ",doubles_found)
                    #print(len(struct)) è 13 --> numero di atomi nell'interfaccia 
    #setdefault() method:argomenti (keyname,value), returns the value of the item with the specified key. 
                    equivalent_structs.setdefault(name, []).append(name_2) #setdefault returna una lista vuota e ci appende name_2
                    doubles_found.append(name_2)

    print(f"Initial number of shifts: {len(hs_u.keys())}") 
    print(f"Number of equivalent shifts: {len(doubles_found)}") 
    for value in equivalent_structs.values():
        for key in value:
            if key in hs_u.keys(): hs_u.pop(key)
            if key in hs_a.keys(): hs_a.pop(key)
    print(f"Number of inequivalent shifts: {len(hs_u.keys())}") 
    return hs_u, hs_a



