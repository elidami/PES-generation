#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
#from tribchem.physics.base.math import rbf_derive_line


# =============================================================================
# EVALUATION OF THE SHEAR STRENGTH
# =============================================================================

class ShearStrength:
    
    @staticmethod
    def get_shear_strength(x, y, rbf, delta=0.01):
        """
        Calculate the shear strength given a path and a potential energy surface.

        Parameters
        ----------
        coords : numpy.ndarray
            Coordinates [x, y] of the path along which you evaluate shear strength.

        rbf : scipy.interpolate.rbf.Rbf
            Contain the information of the interpolation of the potential energy.

        delta : TYPE, optional
            discretized step along x and y for integration. Tuning this value may
            vary slightly the final result. The default is 0.01.

        Returns
        -------
        data_ss_mep : numpy.ndarray
            Profile of potential energy and forces along the MEP.

        ss : float
            The shear strenth along the MEP.

        """

        n = len(x)

        dx = x - np.roll(x,1)
        dy = y - np.roll(y,1)
        dx[0] = 0.
        dy[0] = 0.
        tx = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
        ty = 0.5 * (np.roll(y, -1) - np.roll(y, 1))    
        # potential computed as integral of projection of gradV on string tangent
        Vz = np.zeros(n)
        #derivative of the potential
        x += delta
        tempValp = rbf(np.column_stack([x,y]))  #RBFInterpolator  (metti rbf(x,y) per versione RbF legacy)
        x -= 2.*delta 
        tempValm = rbf(np.column_stack([x,y]))  #RBFInterpolator
        dVx = 0.5*(tempValp - tempValm) / delta
        x += delta
        y += delta
        tempValp = rbf(np.column_stack([x,y]))  #RBFInterpolator
        y -= 2. * delta
        tempValm = rbf(np.column_stack([x,y]))  #RBFInterpolator
        y += delta
        dVy = 0.5 * (tempValp - tempValm) / delta
    
        tforce= - (tx * dVx + ty * dVy)
        force= tforce / np.sqrt(tx**2 + ty**2)
        
        for i in range(n - 1):
            Vz[i + 1] = Vz[i] - 0.5 * (tforce[i] + tforce[i + 1])
            
        Vz -= np.min(Vz)
        #Ve = rbf(x, y)
        Ve = rbf(np.column_stack([x,y])) #RBFInterpolator
        Ve -= np.min(Ve)
        lxy  = np.cumsum(np.sqrt(dx**2 + dy**2))
        data_ss = np.stack((lxy, dVx, dVy, Vz, Ve, force), axis=-1)
        
        ss_min = 10.*np.min(force)
        ss_max = 10.*np.max(force)
        ss = max(abs(ss_min), abs(ss_max))
        
        # TODO : check what is stored in data_ss_mep and keep only important stuff
        return ss, data_ss
    
    @staticmethod
    def get_shear_strength_xy(cell, rbf, params=None):   
        """
        Calculate the shear strength along the x and y directions of the cell.
        Simplified version of GetShearStrength.
        
        TODO : generalize the function in order to calculate the SS along any 
        straight line
        
        """
    
        delta = 0.01
        npoints = 900 #ELISA messo 900 invece di 300
    
        if params != None and isinstance(params, dict):
            for k in params:
                if k == 'delta':
                    delta = params[k]
                elif k == 'npoints':
                    npoints = params[k]   
    
        alat_x = cell[0, 0]
        alat_y = cell[1, 1]
        print(f"alat_x: {alat_x}")
        print(f"alat_y: {alat_y}")
    
        #x = np.arange(-1.5 * alat_x, 1.5 * alat_x, alat_x / npoints)
        #y = np.arange(-1.5 * alat_y, 1.5 * alat_y, alat_y / npoints)  
        #y = y[:-1]
        x = np.linspace(-1.5 * alat_x, 1.5 * alat_x, npoints) #ELISA messo linspace invece di arange
        y = np.linspace(-1.5 * alat_y, 1.5 * alat_y, npoints) #ELISA messo linspace invece di arange
        print(f"len x: {len(x)}")
        print(f"len y: {len(y)}")
        zdev_x = np.zeros(len(x))
        zdev_y = np.zeros(len(y))

        for i in range(len(x)):
            coordx = x[i]
            coordy = y[i]
            zdev_x[i] = rbf_derive_line(rbf, coordx, coordy, m=0, delta=delta)
            zdev_y[i] = rbf_derive_line(rbf, coordx, coordy, m=None, delta=delta)
    
        #Shear strength in GPa    
        ss_x = np.amax(np.abs(zdev_x)) * 10.0
        ss_y = np.amax(np.abs(zdev_y)) * 10.0
        ss_xy = (ss_x, ss_y)
    
        data_ss_xy = np.stack((zdev_x, zdev_y), axis=-1)
    
        return ss_xy, data_ss_xy



import numpy as np
#from geneticalgorithm import geneticalgorithm as ga


def rbf_derive_line(rbf, coordx, coordy, m=None, delta=0.01):
    """
    Calculate the x derivative or the y derivative along a given curve. If m
    is not set as None or zero it calculates the derivative 
    
    Calculate the derivative of a straight line for a (x, y) set of points.

    Parameters
    ----------
    x : np.ndarray or list-like
        Abscissa coordinates.

    y : np.ndarray or list-like
        Ordinate coordinates.

    rbf : function or interpolation object
        Surface function to be used. It can be either a standard function
        or an interpolation object behaving like a function, e.g. an
        interpolation with radial basis function rbf = Rbf(x, y).

    m : str or None, optional
        Type derivative. It can be a float (y = mx line) or None (for making a
        derivative along y). The default is None.

    delta : float, optional
        Discretized step to make the derivative. The default is 0.01.
    
    Returns
    -------
    zdev : float
        Value of the derivative along the straight line.

    """
    
    if m is None:  # Derive along y
        coordx_1 = coordx
        coordx_2 = coordx
        coordy_1 = coordy - delta
        coordy_2 = coordy + delta
    elif m < 1e-10 :  # Derive along x
        coordx_1 = coordx - delta
        coordx_2 = coordx + delta
        coordy_1 = coordy
        coordy_2 = coordy
    else:  # Derive along the straight line with slope m
        coordx_1 = coordx - m * delta 
        coordx_2 = coordx + m * delta
        coordy_1 = coordy
        coordy_2 = coordy
    
    # Calculate the derivative
    #V_1 = rbf(coordx_1, coordy_1)
    #V_2 = rbf(coordx_2, coordy_2)

    #RBFInterpolator
    V_1 = rbf(np.column_stack([coordx_1, coordy_1]))
    V_2 = rbf(np.column_stack([coordx_2, coordy_2]))
    zdev = 0.5 * (V_2 - V_1) / delta

    return zdev

