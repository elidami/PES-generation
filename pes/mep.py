#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import os

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#rom tasktools import read_default_params
from shearstrength import ShearStrength
from shearstrength import rbf_derive_line
from solidstate import zfill_cell


#currentdir = os.path.dirname(__file__)


class MEP:
    """
    Instanciate a solver to calculate the minimum path of a 3D surface created
    by a f(x, y) functions. The MEP will link the absolute and local minima of
    the surface passing through the saddle points. The intuitive idea is that
    you are moving between the valleys of a mountainous landscape.

    """
    
    #defaults = currentdir + '/defaults_tribo.json'
    method = ['zerotemp']
    optimization = [None, 'x', 'y', 'bs_line']

    def __init__(self, rbf, cell):
        """
        Initialize a MEP solver object, which is capable of calculating the
        minimum energy path on top of a three dimensional surface by means of
        different techniques. The rbf object passed to the constructor method
        should be a two dimensional function, i.e. f(x, y) and it is used to
        calculate the value assumed on the three dimensional landscape in
        different positions.

        Parameters
        ----------
        rbf : function or interpolation object
            Surface function to be used. It can be either a standard function
            or an interpolation object behaving like a function, e.g. an
            interpolation with radial basis function rbf = Rbf(x, y).

        cell : list or np.ndarray
            Cell that will contain the string to calculate the MEP on top of
            the surface f(x, y). If the rbf object comes from an interpolation,
            you should be sure that the cell is well inside the interpolation
            region and not on the edges.

        """

        self.rbf = rbf
        self.cell = cell

    def get_mep(self, method='zerotemp', optimization='bs_line', **kwargs):
        """
        Wrapper to start the MEP calculation. See __get_mep for documentation.

        """
        
        # Calculate the Minimum Energy Path
        data = self.__get_mep(method=method, optimization=optimization, **kwargs)
        
        # Save the attributes of the MEP
        self.mep = data[0]
        self.mep_data = {
            "nstep": data[1][0],
            "tol": data[1][1]
            }

        # Store data concerning the starting string to get the MEP
        values = data[2]
        self.bsmep = np.column_stack((values[0], values[1]))
        self.bsmep_data = {
            "shear_strength": values[2],
            }
        if optimization == 'bs_line':
            self.bsmep_data.update({'theta': values[2][1]})

        return self.mep, self.mep_data

    def get_shear_strenght(self, delta=0.01):
        """
        Calculate the shear strength along the MEP and save it as attribute.

        Parameters
        ----------
        delta : float, optional
            Discretized step for making the derivative. The default is 0.01.

        Raises
        ------
        ValueError
            If the MEP has not been previously calculated.

        """

        try:
            shear, data = self.__get_shear_strength(self.mep[:, 0], self.mep[:, 1], 
                                                    self.rbf, delta)
        except:
            raise ValueError('The MEP is not calculated properly. Run get_mep '
                             'before calculating the shear strength')

        # Update the mep data
        self.mep_data.update({"shear_strength": shear,
                              "lxy": data[0],
                              "dvx": data[1],
                              "dvy": data[2],
                              "vz": data[3],
                              "ve": data[4],
                              "force": data[5]})
        return shear, data

    def __get_shear_strength(self, x, y, rbf, delta=0.01):
        return ShearStrength.get_shear_strength(x, y, rbf, delta)

    def __get_mep(self, method='zerotemp', optimization='bs_line', **kwargs):

        if method is None or method == 'zerotemp':
            data = MEP.zerotemp_method(self.rbf, self.cell, optimization, **kwargs)
        else:
            raise ValueError('Error in selecting the method for calculating MEP. '
                             'At the moment only "zerotemp" is implemented.')

        return data

    @staticmethod
    def zerotemp_method(rbf, cell, optimization='bs_line', **kwargs):
        """
        Calculate the Minimum Energy Path on top of a surface f(x, y) using
        the improved simplified string method, base on the zero temperature 
        string method [1, 2]. This is a static method and can be used outside
        the 
    
        Parameters
        ----------        
        rbf : scipy.interpolate.rbf.Rbf
            Interpolation function to calculate the surface potential energy.
        
        cell : numpy.ndarray
            Vectors of the unit lattice cell containing the studied structure.
            
        optimization : str or None, optional
            Select or build an improved starting string to converge faster.
            Allowed values are stored in the optimization class attribute. If
            nothing is passed, an horizontal string will be selected, i.e.
            oriented as the x-axis of the cell. The default value is None.
            
        **kwargs : optional
            Computational parameters needed to the algorightm computing the MEP. 
            The default is None. If left to None, the following values are used:
            
            npts = 101
            extent = [1.5, 1.5]
            h = 0.001
            nstepmax = 99999
            tol = 1e-7
            delta = 0.01
            
            npts : Number of points constituting the string.
            extent : String spatial extension for x and y (build a rectangle).
            h : time-step, limited by ODE step but independent from npts
            nstepmax : max number of iteration to get the MEP to converge
            tol : tolerance for the convergence of the MEP
            delta : discretized step for integration along x and y
            
            You can change some or any of these values passing them as kwargs

        Returns
        -------
        mep : numpy.dnarray
            Set of coordinates along the MEP [x, y]. Run rbf(mep) to see the 
            potential energy profile along the MEP.
            
        mep_convergency : tuple
            Contains information about the algorithm convergence, i.e. (nstep, tol)
            nstep : number of steps done by the algorithm
            tol : final difference between the points of the string
            
        References
        ----------
        [1] W. E, W. Ren, E. Vanden-Eijnden, String method for the study of 
        rare events, Phys.Rev. B 66 (2002) 052301, https://doi.org/10.1103/PhysRevB.66.052301.
        [2] W. E, W. Ren, E. Vanden-Eijnden, Simplified and improved string 
        method for computing the minimum energy paths in barrier-crossing 
        events, J. Chem. Phys. 126 (16) (2007) 164103, https://doi.org/10.1063/1.2720838.

        """
        
        # Initialize the parameters for the computation
        #p = read_default_params(MEP.defaults, 'zerotemp', kwargs)
        #npts, nstepmax, h, extent, delta, tol = p['npts'], p['nstepmax'], p['h'], p['extent'], p['delta'], p['tol']
        
        #ELISA
        npts = 101
        extent = [1.5, 1.5]
        h = 0.001
        nstepmax = 99999
        tol = 1e-7
        delta = 0.01
        
        # Calculate the initial string
        data = initialize_string(cell=cell, rbf=rbf, optimization=optimization, npts=npts,
                                    extent=extent, delta=delta)
        x, y = data[:2]
        g = np.linspace(0, 1, npts)
        
        # To make everything quicker, use rbf to generate a smaller mesh with a smaller delta x, y,
        # with griddata generate the derivative. Gradients on a mesh then should be only interpolated.

        # initial parametrization  
        dx = x - np.roll(x, 1) #x(i)-x(i-1)
        dy = y - np.roll(y, 1)
        dx[0] = 0 #i dx e dy non sono uguali tra loro perchè la stringa in fase di inizializzazione è stata randomizzata
        dy[0] = 0
        lxy  = np.cumsum(np.sqrt(dx**2 + dy**2))        
        lxy /= lxy[npts - 1]
        xf = interp1d(lxy, x, kind='cubic')
        x  =  xf(g)
        yf = interp1d(lxy, y, kind='cubic')
        y  =  yf(g)
        
        de = delta * np.ones(npts)
         
        # Main loop
        for nstep in range(int(nstepmax)):
            if nstep ==1000:
                data_to_file = np.column_stack((x, y))
                # Salva l'array in un file CSV
                np.savetxt("prima.csv", data_to_file, delimiter=" ")
            # Calculation of the x and y-components of the force.
            # dVx and dVy are the derivative of the potential

            #tempValp = rbf(x + de, y)
            #tempValm = rbf(x - de, y)
            #RBFInterpolator
            tempValp = rbf(np.column_stack([x + de, y]))
            tempValm = rbf(np.column_stack([x - de, y]))
            dVx = 0.5 * (tempValp - tempValm) / delta
            
            #tempValp = rbf(x, y + de)
            #tempValm = rbf(x, y - de)
            #RBFInterpolator
            tempValp = rbf(np.column_stack([x, y + de]))
            tempValm = rbf(np.column_stack([x, y - de]))
            dVy = 0.5 * (tempValp - tempValm) / delta

            # String steps:
            # 1. Evolve
            x0 = x.copy()
            y0 = y.copy()
            x -= h * dVx
            y -= h * dVy

            # 2. Reparametrize  
            dx = x - np.roll(x, 1)
            dy = y - np.roll(y, 1)
            dx[0], dy[0] = 0., 0.
            lxy  = np.cumsum(np.sqrt(dx**2 + dy**2))
            lxy /= lxy[npts - 1]
            xf = interp1d(lxy, x, kind='cubic')
            x  =  xf(g)
            yf = interp1d(lxy, y, kind='cubic')
            y  =  yf(g)
            t = (np.linalg.norm(x - x0) + np.linalg.norm(y - y0)) / npts #distanza media tra i punti
            if nstep ==1000:
                data_to_file = np.column_stack((x, y))
                # Salva l'array in un file CSV
                np.savetxt("dopo.csv", data_to_file, delimiter=" ")
            if t <= tol:
               break

        mep = np.column_stack([x, y])
        mep_convergence = (nstep, t)
        print("nstep: ", nstep)
        return mep, mep_convergence, data


# =============================================================================
# Tools to evaluate the best starting string to calculate the MEP
# =============================================================================

def initialize_string(cell, rbf, optimization=None, npts=101, 
                      extent=[1.5, 1.5], delta=0.01):
    """
    Generate the inital string to calculate the MEP

    """

    # TODO: Generalize to any type of 2D cell
    # WARNING: Works only with squared lattice for now
    alat_x = zfill_cell(cell)[0, 0]
    alat_y = zfill_cell(cell)[1, 1]
    
    # Define the initial string
    xlim = (-extent[0] * alat_x, extent[0] * alat_x)
    ylim = (-extent[1] * alat_y, extent[1] * alat_y)
    g = np.linspace(0, 1, npts)
    
    data = None
    if optimization == 'x' or optimization is None:  # x-axis
        x = (xlim[1] - xlim[0]) * g + xlim[0]
        y = np.zeros(npts) + 0.05 * alat_y * np.sin(2.*np.pi*g)
    elif optimization == 'y':  # y-axis
        x = np.zeros(npts) + 0.05 * alat_x * np.sin(2.*np.pi*g)
        y = (ylim[1] - ylim[0]) * g + ylim[0]
    elif optimization == 'bs_line':
        x = (xlim[1] - xlim[0]) * g + xlim[0]
        y = np.zeros(npts)
        x, y, ss, theta = get_bs_line(cell, rbf, npts, extent, 1.0, delta)
        data = (ss, theta)
    else:
        raise ValueError('Wrong optimization provided')
    
    return x, y, data

def get_bs_line(cell, rbf, npts=101, extent=[1.5, 1.5], delta_theta=1,
                delta=0.01):
    """
    Evaluate the best starting linear string to start MEP calculation.

    Parameters
    ----------
    cell : np.ndarray
        Unit lattice cell of the structure.

    rbf : scipy.interpolate.rbf.Rbf
        Contain the information of the interpolation of the potential energy.
    
    delta_theta : float or int, optional
        Angle interval (degrees) to calculate the best starting line string.
        The default is 1.
    
    delta : float or int, optional
        Discretized step along x and y for integration. The default is 0.01

    Returns
    -------
    bsmep : TYPE
        DESCRIPTION.

    ss_bs_line : TYPE
        Shear strength calculated along the string. Units are GPa

    theta : float
        Best orientation for starting calculating the MEP. Units are radiants

    """    
    
    # Define the initial horizontal string
    x, _, _ = initialize_string(cell, rbf, 'x', npts, extent, delta)
    
    delta_theta *= np.pi / 180  # Convert the angle to radiants
    
    # Set parameters needed in main loop
    n = len(x)
    g = np.linspace(0, 1, n)
    xlim = zfill_cell(cell)[0, 0] * extent[0] * np.array([-1, 1])
    ylim = zfill_cell(cell)[1, 1] * extent[1] * np.array([-1, 1])
    
    # Calculate the shear strength for all the intervals, starting from x and y
    data_ss = []
    data_th = []
    ss_xy, _ = ShearStrength.get_shear_strength_xy(zfill_cell(cell), rbf)
    data_ss.extend((float(ss_xy[0]), float(ss_xy[1])))
    data_th.extend((0, np.pi/2))
    
    check_pts = int(np.pi / delta_theta)
    theta = 0
    i = 0
    while i < check_pts:
        theta += delta_theta
        
        # Calculate the derivative only for non-degenerate strings, i.e. y-axis
        if not abs(theta - np.pi / 2) < 1e-10:
            m = np.tan(theta)
            y = m * x.copy()                       

            zdev = np.zeros(n)        
            for j in range(n):
                coordx = x[j]
                coordy = y[j]
                zdev[j] = rbf_derive_line(rbf, coordx, coordy, m, delta)        
            #Shear strength in GPa
            ss = np.amax(np.abs(zdev)) * 10.0

            data_th.append(theta)
            data_ss.append(float(ss))
        i += 1

    index = data_ss.index(np.amin(np.abs(data_ss)))
    theta = data_th[index]
    ss = data_ss[index]

    if abs(theta) == np.pi/2:
        x = 0.01*(xlim[1] - xlim[0]) * np.sin(2.*np.pi*g) # np.zeros(n)
        y = (ylim[1] - ylim[0]) * g + ylim[0]
    elif abs(theta) < 1e-10:
        x = (xlim[1] - xlim[0]) * g + xlim[0]
        y = 0.01*(ylim[1] - ylim[0]) * np.sin(2.*np.pi*g) # np.zeros(n)
    else:
        x = (xlim[1] - xlim[0]) * g + xlim[0]
        y = np.tan(theta) * x.copy()
    
    # Tiny perturbation of the string
    x += np.random.randn(len(x)) * (xlim[1] - xlim[0]) / 10000
    y += np.random.randn(len(y)) * (ylim[1] - ylim[0]) / 10000
    
    return x, y, ss, theta
