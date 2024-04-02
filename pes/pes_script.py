#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Manage the calculation of the Potential Energy Surface (PES) through an high
level class interface. A PES object can be created and the Minimum Energy Path 
(MEP) and the Shear Strength can be evaluated.

This module contains:
    - PES
        Interpolate a 3D Surface with radial basis functions and calculate the
        less stipest path on top of it.
        
        Methods:
        - rbf
        - make_pes
        - __make_rbf
        - __infer_theta
        - to_file
        - from_file
        - from_hs
        - plot

    - PESModel

    - unfold_pes
        Unfold the energies calculated for the unique HS points of an 
        interface/surface, associating them to the replicated HS points 
        covering the whole surface of the cell.


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

Version 1.2.1
-------------------------------------------------------------------------------
- Slightly improve the PES plotting, now make a symmetrized plot around zero
- Add a distinction between plot_name and fig_name in PES plotting

Version 1.2.0
-------------------------------------------------------------------------------
- Add a class to store the PES mathematical models. Contains the methods:
    - v, ci, u
- Add the AnalyticPES class, which initialize a PES using mathematical models.
  Contains the methods and properties:
    - __make_fit, make_pes, v, corrugation, plot_ppes, plot

Version 1.1.0
-------------------------------------------------------------------------------
- Bug Fix: now the code of the PES works fine
- Implement correctly the orthorombize features when making the PES
- Implement the rotation of the provided lattice cell and a method that can
  infer the correct angle. This is the only way to keep the geometry and
  orthorombize correctly an hexagonal lattice cell
- Add a method to plot the initial cell and points

Version 1.0.0
-------------------------------------------------------------------------------
- First module release, contains class: PES, functions: unfold_pes

"""

import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

from mep import MEP
from solidstate import (
    pbc_coordinates,
    rm_duplicates_2d,
    replicate_2d,
    generate_uniform_grid,
    zfill_cell,
    orthorombize_2d,
    rotate
)


class PES(MEP):
    """
    This class takes a data grid of positions and values, distributed within a
    certain lattice cell. The PES is obtained by replicating the points, and 
    then interpolating the given values with radial basis functions.
    It is also possible to orthorombize the cell before interpolation, however
    in this case it is necessary to ensure that the cell is initially correctly
    oriented, so that the periodicity is granted before the interpolation.

    """

    def __init__(self, data, cell):
        """
        Initialize the class, by passing the data to be interpolated, as a 
        Nx3 matrix, and the lattice cell of the system, as a 3x3 or 2x2 matrix.
        
        The input argument `data` should have the form:
            x1 y1 E1
            x2 y2 E2
             .  .  .
             .  .  .
             .  .  .

        The third column can have whatever units, e.g. in the case of energy you
        might use eV or J/m^2. However, the first two columns need to be Ansgtrom.
        
        The input argument `cell` should have the form:
            x1 y1 z1
            x2 y2 z2
            x3 y3 z3
        or:
            x1 y1
            x2 y2
        
        The lattice cell coordinates need to be Angstrom. If a 3x3 matrix is 
        provided, only the 2x2 planar inner matrix is considered to build the
        PES, being it a two-dimensional function, i.e. f(x, y).
        When a 2x2 matrix is provided, the cell is still filled to a 3x3 with 
        default numbers, to keep the informatic consistency of the code.

        Parameters
        ----------
        data : numpy.array
            Data grid, with shape Nx3, containing all the cartesian displacements
            or coordinates on the first two columns and the values to be
            interpolated on the third column.

        cell : numpy.array
            Lattice cell of the system, as a 2x2 matrix in Angstrom units.

        """

        self.data = data
        self.cell = zfill_cell(cell)

    def make_pes(self, replicate_of=(2, 2), orthorombize=False, theta=None,
                 density=20, tol=1e-4, set_emin_xy=True):
        """
        Parameters
        ----------
        replicate_of : TYPE, optional
            DESCRIPTION. The default is (3, 3).
        function : TYPE, optional
            DESCRIPTION. The default is 'cubic'.
        density : TYPE, optional
            DESCRIPTION. The default is 20.

        """

        output = self.__make_rbf(self.data, self.cell, replicate_of, 
                                 orthorombize, theta, density, tol, set_emin_xy)
        self._rbf, self.pes, self.cell_2d = output

    def __make_rbf(self, data, cell, replicate_of=(2, 2), orthorombize=False,
                   theta=None, density=20, tol=1e-4, set_emin_xy=True):
        """
        Effective calculation of the PES by means of radial basis function
        interpolation. It is wrapped by the `make_pes` method.

        """

        # Rotate the cell in order to have it with the diagonal // to x
        if bool(theta):
            if not type(theta) in (int, float):
                theta = self.__infer_theta()
            fake_d = np.column_stack((data[:, :2], np.zeros(data.shape[0])))
            data[:, :2] = rotate(fake_d, mod='z', theta=theta)[:, :2]
            cell = rotate(cell, mod='z', theta=theta)

        # Apply pbc and make the cell to be orthorombic
        if orthorombize:
           data, cell_2d = orthorombize_2d(data, cell, tol, to_plot=False)
        else:
           cell_2d = cell[:2, :2]

        # Be sure points are not represented twice by ensuring rows in data are 
        # unique. Check also that x and y coordinates are inside the unit cell.
        fake_cell = np.vstack((np.column_stack((cell_2d, [0, 0])), [0, 0, 1]))
        data[:, :2] = pbc_coordinates(data[:, :2], fake_cell, to_array=True)
        data = rm_duplicates_2d(data)

        # Scale the energies by setting the minimum to zero
        emin = min(data[:, 2])
        if set_emin_xy:  # Set also the zero to zero
            xmin, ymin = data[np.argwhere(data[:, 2] == emin)][0][0][:2] 
            data[:, 0] -= xmin
            data[:, 1] -= ymin
        data[:, 2] -= emin

        # Replicate points and interpolate
        #import os
        #print(os.environ['OMP_NUM_THREADS'])
        data, _ = replicate_2d(data, cell_2d, replicate_of, symm=True)
        data = rm_duplicates_2d(data)
        print('length data of interpolation:', len(data))
        #rbf = Rbf(data[:, 0], data[:, 1], data[:, 2], function='cubic')

        #implementazione con RBFInterpolator
        from scipy.interpolate import RBFInterpolator
        rbf = RBFInterpolator(data[:, :2], data[:, 2], kernel='cubic')


        # Calculate the PES on a very dense and uniform grid. Useful for further 
        # analysis (MEP, shear strength) and to plot the PES
        if density is not None:
            coordinates = generate_uniform_grid(cell, density, to_plot=False)
            #energy= rbf(coordinates[:, 0], coordinates[:, 1])
            energy= rbf(coordinates[:, :2]) #RBFInterpolator
            pes = np.column_stack([coordinates[:, :2], energy])
        else:
            pes = data

        return rbf, pes, cell_2d

    def __infer_theta(self, tol=1e-4):

        a = np.linalg.norm(self.cell[0, :2])
        b = np.linalg.norm(self.cell[1, :2])
        
        alfa = 180 / np.pi * np.arccos(np.dot(self.cell[0, :2], [1, 0]) / a)
        beta = 180 / np.pi * np.arccos(np.dot(self.cell[1, :2], [1, 0]) / b)
        if alfa > 90:
            alfa = 180 - alfa
        if beta > 90:
            beta = 180 - beta

        if not abs(alfa - beta) < tol:
            theta = (alfa + beta) / 2.
        else:
            theta = 0.

        return theta
    
    #def rbf(self, x, y):
        #return self._rbf(x, y)
    
    #RBFInterpolator
    def rbf(self, x):
        return self._rbf(x)
    
    def plot_cell(self, replicate_of=(1, 1), is_2d=False):
        
        if is_2d:
            cell = self.cell_2d.copy()
        else:
            cell = self.cell.copy()
        
        data, _ = replicate_2d(self.data, cell, replicate_of)
        
        plt.plot([0, cell[0, 0], cell[0, 0]+cell[1, 0], cell[1, 0], 0], 
             [0, cell[0, 1], cell[0, 1]+cell[1, 1], cell[1, 1], 0])
        plt.plot(data[:, 0], data[:, 1], 'o')
        plt.show()

    def to_file(self, fname='pes.dat'):
        """
        Save the pes data to file.

        """
        np.savetxt(fname, self.pes)

    def plot(self, extent=(1, 1), mpts=(200j, 200j), figsize=(7, 7), symm=False,
             level=40, shrink=1, mep=None, show_grid=False, title=None, 
             add_axis=None, to_fig=None, colorbar_limit=None):
        """
        Plot the PES and eventually save it

        """

        # Create a mesh grid for plotting
        a = np.sum(np.abs(self.cell_2d[:, 0]))
        b = np.sum(np.abs(self.cell_2d[:, 1]))

        mptx, mpty = (mpts, mpts) if isinstance(mpts, complex) else mpts

        x, y = np.mgrid[- extent[0] * a * int(symm) : extent[0] * a : mptx,
                        - extent[1] * b * int(symm) : extent[1] * b : mpty]
        #z = self.rbf(x, y)
        #RBFInterpolator
        z = self.rbf(np.column_stack([x.ravel(), y.ravel()]))
        # Reshape z to have the same shape as x and y
        z = z.reshape(x.shape)

        z -= np.min(z)

        # Create the plot
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        anglerot='vertical'
        #zt1=plt.contourf(x, y, z, level, cmap=plt.cm.RdYlBu_r)
        #zt1.set_clim(vmin=0, vmax=colorbar_limit) # ELISA: added to set the scale of the colorbar

        # Normalizza la colorbar (ELISA): modifica per la scala colori del grafico PES
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=colorbar_limit)
        zt1=plt.contourf(x, y, z, level, cmap=plt.cm.RdYlBu_r)
        zt1.set_clim(vmin=0, vmax=colorbar_limit)

        # Add legend center the plot and write axis labels
        ax.axis([-extent[0] * a * int(symm), extent[0] * a, 
                 -extent[1] * b * int(symm), extent[1] * b])
        plt.xlabel(r"distance ($\AA$)", fontsize=12, family='serif')
        plt.ylabel(r"distance ($\AA$)", fontsize=12, family='serif')
        cbar1=plt.colorbar(zt1, ax=ax, orientation=anglerot, shrink=shrink)
        cbar1.set_label(r'$E_{adh} (J/m^2)$', rotation=270, labelpad=20,
                        fontsize=15, family='serif')
        
        if mep is not None:
            mep = np.array(mep)  # Assicurati che mep sia un array numpy

            min_x = mep[:, 0].min()
            min_y = mep[:, 1].min()

            mep[:, 0] -= min_x
            mep[:, 1] -= min_y -1.5
            ax.plot(mep[:, 0], mep[:, 1], '.-', c='black', ms=2)
            
        if title is not None:
            plt.title("PES for " + str(title), fontsize=18, family='serif')
        
        if show_grid:
            plt.plot(self.data[:, 0], self.data[:, 1], 'o')
        
        # Add axis to the origin, to indicate the two crystalline orientations
        # in the plane (along cartesian x and y).
        if add_axis is not None:
            ax.plot(0.,0., 'w.', ms=7)
            ax.quiver(0, 0, 1, 0, scale=1., scale_units='inches', width=0.01, 
                      color='white')
            ax.quiver(0, 0, 0, 1, scale=1., scale_units='inches', width=0.01, 
                      color='white')
            ax.text(0.25, -0.5, str(add_axis[0]).replace(',', '').strip(), 
                    rotation='horizontal', color='white', fontsize=14)
            ax.text(-0.5, 0.25, str(add_axis[1]).replace(',', '').strip(), 
                    rotation='vertical', color='white', fontsize=14)
        
        # Small bug in matplotlib, improve the plot lines and graphics
        for zt1 in zt1.collections:
            zt1.set_edgecolor("face")
            zt1.set_linewidth(0.000000000001)
        
        # Save the figure
        if to_fig is not None:
            plt.savefig(str(to_fig), dpi=300)
        
        self.pes_plot = fig
    
    def calculate_mep(self, starting_point=None):
        pass

    @classmethod
    def from_file(cls, fname, cell):
        """
        Generate the pes out of an existing set of data which has

        """
        pes = np.genfromtxt(fname, dtype=float)
        return cls(pes, cell)

    @classmethod
    def from_hs(cls, hs, energy, cell, to_fig=None, point_density=20):
        """
        Create a PES object out of a list of high symmetry points and energies.
        
        Main function to get the Potential Energy Surface (PES) for an interface. 
        The points are replicated to span a 3x3 lattice cell and are interpolated
        by using Radial Basis Functions (cubic function).
        In the output data the energy is normalized so that the absolute minimum
        is 0. Furthermore it is made sure that the lateral point are inside the
        unit cell before they are replicated.

        The entire set of HS points covering the interface associated with 
        the corresponding energy.
                x[0]  y[0]  E[0]
                x[1]  y[1]  E[1]
                .     .     .
                .     .     .
                .     .     .
        
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
        
        # Unfold the PES points and create the data array to feed the PES class
        _, data = unfold_pes(hs, energy)
        return cls(data, cell)



# =============================================================================
# Tools for the PES
# =============================================================================

def unfold_pes(hs_u, energy_u):
    """
    Unfold the energies calculated for the unique HS points of an interface,
    associating them to the replicated HS points covering the whole surface of 
    the cell. It is supposed that you provide a set of unique HS points and the
    corresponding energies, to be replicated to full the entire lattice cell.

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
        interpolate a PES. It's E_list without labels and with array type. 
        The structure of the matrix is:
            
            np.array([ [x_1, y_1, E_1], 
                       [x_2, y_2, E_2], 
                       ...             ])

    """

    # Initialize lists for the result
    e_list = []
    e_array = []

    # Extract the element
    for element in energy_u:
       label  = element[0]
       energy = element[3]

       # Associate each Energy to all the corresponding HS values
       for row in hs_u[label]:
          x_shift = row[0]
          y_shift = row[1]

          e_list.append([label, x_shift, y_shift, energy])
          e_array.append([x_shift, y_shift, energy])

    e_array = np.array(e_array)      
    
    return e_list, e_array
