import numpy as np
from solidstate import pbc_coordinates
from pes import PES

# =============================================================================
# EVALUATION OF THE PES - MAIN
# =============================================================================


def get_pes(hs_all, E, cell, density=20, title=None, to_fig=None, colorbar_limit=None):
    """
    Interpolate the PES using a list of high symmetry points and energies.
    
    Main function to get the Potential Energy Surface (PES) for an interface. 
    The points are replicated to span a 3x3 lattice cell and are interpolated
    by using Radial Basis Functions (cubic function).
    In the output data the energy is normalized so that the absolute minimum
    is 0. Furthermore it is made sure that the lateral point are inside the
    unit cell before they are replicated.

    Parameters
    ----------       
    hs : dict
        Unfolded HS points of the interface, covering all the surface.
        
    E : dict
        Contains the energy calculated for each unique surfacial HS site.
    
    cell : np.ndarray
        Vectors of the lattice cell of the interface.
        
    to_fig : string, optional CURRENTLY NOT IMPLEMENTED
        Name of the image that you want to save, it will be: 'to_fig'+'.pdf' 
        Suggested name: name = 'PES_' + 'Name of the interface'.
        The default is None and no image is saved.

    Returns
    -------
    rbf : scipy.interpolate.rbf.Rbf
        Object containing the information of the interpolation of the potential
        energy. Call it on a set of [x, y] coordinates to obtain the energy 
        values for those points. Usage: rbf([x, y])
        
    pes_dict : dict
        Dictionary containing the points and the associated energies divided
        by type. Useful to keep track of the type of the HS sites and their
        energies. To be saved to DB (?)
    
    pes_data : np.ndarray
        The entire set of HS points covering the interface with the 
        corresponding energies.
        Format:
            x[0]  y[0]  E[0]
            x[1]  y[1]  E[1]
             .     .     .
             .     .     .
             .     .     .
             
    """
    
    # Unfold the PES points
    v_list, data = unfold_pes(hs_all, E)

    #making sure points are not represented twice by ensuring rows in data are unique
    data = remove_duplicates(data)
    
    #make sure that the x and y coordinates are inside the unit cell.
    x_y_insideCell = pbc_coordinates(data[:, :2], cell, to_array=True)

    data[:, :2] = x_y_insideCell
        
    # Normalize the minimum to 0
    data[:,2] = (data[:,2] - min(data[:,2]))
    
    # Interpolate the data with Radial Basis Function
    # data_rep = replicate(data, cell, replicate_of=(3, 3) )
    solver = PES(data, cell)
    solver.make_pes(replicate_of=(7, 7), density=20, tol=1e-4) #OMAR
    #solver.make_pes(replicate_of=(3, 3), density=20, tol=1e-4) #ELISA
    
    # Make the plot
    if to_fig is not None:
        #solver.plot(extent=(2, 2), mpts=(200j, 200j), title=title, to_fig=to_fig) #OMAR
        solver.plot(extent=(3, 3), mpts=(200j, 200j), title=title, to_fig=to_fig, colorbar_limit=colorbar_limit) #ELISA
    
    return solver, v_list, data


# =============================================================================
# UTILITY FOR THE PES
# =============================================================================

def remove_duplicates(data, rounding_decimal=5):
    """
    Remove duplicated points from the dict of the PES.
    
    """
        
    xy = np.round(data[:, :2], decimals=rounding_decimal)
    E_dict = {}
    for i in xy:
        duplicates = np.where((xy == i).all(axis=1))[0]
        E_list = []
        for j in duplicates:
            E_list.append(data[j,2])
        E_dict[str(i)]=np.mean(E_list)
    xy_unique = np.unique(xy, axis=0)
    E=[]
    for i in xy_unique:
        E.append([E_dict[str(i)]])
    
    return np.hstack((xy_unique, np.array(E)))
    
    
def unfold_pes(hs_all, E_unique):
    """
    Unfold the energies calculated for the unique HS points of an interface,
    associating them to the replicated HS points covering the whole surface
    cell. hs_all is a dictionary, E_unique a list.

    Parameters
    ----------
        
    hs_all : dict
        Surfacial HS sites that has been unfolded (replicated) across the whole
        lattice cell of slab. 
        Ex. To the key 'ontop_1 + bridge_1' will correspond n points, spanning
        the entire plane axb of the lattice cell. Data is a (n, 3) numpy array.
    
    E : list
        Contains the energy calculated for each unique interfacial HS site.
        The energy are calculated by means of ab initio simulations (VASP).
        E must have the following structure:
            
            [ [label_1, x_1, y_1, E_1], 
              [label_2, x_2, y_2, E_2], 
              ...                      ]
        
        Ex. label should corresponds to the keys in hs_all, associated to a 
        certain shit between the lower and upper slab, e.g. 'ontop_1+bridge_1'.
        
    Returns
    -------
    E_list : list
        It's basically the same as E_unique but contains all the HS points 
        replicated on the whole interface. The structure of the list is:
            
            [ [label_1, x_1, y_1, E_1], 
              [label_2, x_2, y_2, E_2], 
              ...                      ]
        
    E_array : np.ndarray
        Numpy matrix containing the coordinates and the energy useful to 
        interpolate the PES. It's E_list without labels and with array type. 
        The structure of the matrix is:
            
            np.array([ [x_1, y_1, E_1], 
                       [x_2, y_2, E_2], 
                       ...             ])

    """

    # Initialize lists for the result
    E_list = []
    E_array = []
    
    # Extract the element
    for element in E_unique:
       label  = element[0]
       energy = element[3]
       
       # Associate each Energy to all the corresponding HS values
       for row in hs_all[label]:
          x_shift = row[0]
          y_shift = row[1]
          
          E_list.append([label, x_shift, y_shift, energy])
          E_array.append([x_shift, y_shift, energy])
          
    E_array = np.array(E_array)      
    
    return E_list, E_array
