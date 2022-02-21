import numpy as np
from numpy import ogrid
import pandas as pd

def SCA_proj(R0, r_crop, i, j, D_output=128, base_img_center = None):
    """
    Computes the mask of the Sun-Centered Angular projection.
    
    Principle
    ----------

    Principle of the SCA projection :
     - The x axis depicts the angular distance to the sun
     - The y axis depicts the angle between the direction from the sun and the vector (0,-1)

    Parameters
    ----------
    R0 : int
        Radius of the hemispherical image to be projected (in pixels)
    r_crop : float
        Fraction of the radius R0 to be covered by the projection 
    (i, j) : int, int
        Coordinates of the sun
    D_output : int, optional
        Size of the projection
    base_img_center : (int, int), optional
        Coordinates of the center of the hemispherical image to be projected

    Returns
    -------
    (coordsproj_i, coordsproj_j)

    coordsproj_i : np.ndarray of shape (D_output, D_output) and dtype int
    coordsproj_j : np.ndarray of shape (D_output, D_output) and dtype int
        (coordsproj_i[k,l], coordsproj_j[k,l]) are the coordinates of the point which, projected, would give the point (k,l)
    """

    if base_img_center is None:
        y0 = 480
        x0 = 640
    else:
        y0, x0 = base_img_center

    lamda = np.pi*((i/R0-1)**2 + (j/R0-1)**2)**0.5/2
    if i == R0:
        theta = np.sign(j-R0)*np.pi/2
    else:
        theta = np.arctan((j-R0)/(i-R0))
    
    coords_proj = (np.zeros((D_output,D_output), dtype = int),np.zeros((D_output,D_output), dtype = int))
    for r_i, r in enumerate(np.linspace(0,1, D_output, endpoint=False)*r_crop):
        for th_i, th in enumerate(np.linspace(0,2*np.pi, D_output, endpoint=False)):
            l2 = np.pi*r/2
            rel_l = np.arccos(np.cos(lamda)*np.cos(l2)-np.sin(lamda)*np.sin(l2)*np.cos(th))
            rel_r = 2*rel_l/np.pi
            K = (np.sin(l2)*np.cos(th)*np.cos(lamda)+np.cos(l2)*np.sin(lamda))
            rel_t = np.arctan(np.sin(l2)*np.sin(th)/K) + (1 - np.sign(K/np.sin(rel_l)))*np.pi/2 + theta
            try:
                i0 = int(np.cos(rel_t)*R0*rel_r + y0)
                j0 = int(np.sin(rel_t)*R0*rel_r + x0)
            except:
                i0 = 0
                j0 = 0 
            coords_proj[0][th_i,r_i] = i0
            coords_proj[1][th_i,r_i] = j0
    
    return coords_proj

def planar_proj(R0, h, D_output=128, base_img_center = None):
    """
    Computes the mask of the planar projection.
    
    Principle
    ----------

    Principle of the planar projection : Each point of the hemispheric image is projected 
    on a plane orthogonal to the zenith of the observer, and located at a certain altitude 
    H from the position of the observer.

    Parameters
    ----------
    R0 : int
        Radius of the hemispherical image to be projected (in pixels)
    h : float
        Ratio between the altitude of the plane and the size of the plane
    D_output : int, optional
        Size of the projection
    base_img_center : (int, int), optional
        Coordinates of the center of the hemispherical image to be projected

    Returns
    -------
    (coordsproj_i, coordsproj_j)

    coordsproj_i : np.ndarray of shape (D_output, D_output) and dtype int
    coordsproj_j : np.ndarray of shape (D_output, D_output) and dtype int
        (coordsproj_i[k,l], coordsproj_j[k,l]) are the oordinates of the point which, projected, would give the point (k,l)
    """

    if base_img_center is None:
        y0 = 480
        x0 = 640
    else:
        y0, x0 = base_img_center

    coords_proj = (np.zeros((D_output,D_output), dtype = int),np.zeros((D_output,D_output), dtype = int))

    for i in range(D_output):
        for j in range(D_output):
            if i != D_output//2 or j != D_output//2:
                r2 = ((i/D_output-1/2)**2 + (j/D_output-1/2)**2)**0.5
                r0 = 2*np.arctan(r2/h)/np.pi # (1/(1+h**2/r2**2))**0.5
                cost = (i/D_output-1/2)/r2
                sint = (j/D_output-1/2)/r2
                i0 = int(cost*r0*R0 + y0)
                j0 = int(sint*r0*R0 + x0)
                coords_proj[0][i,j] = i0
                coords_proj[1][i,j] = j0
            else:
                coords_proj[0][i,j] = y0
                coords_proj[1][i,j] = x0            
       
    return coords_proj

def inv_planar_proj(R0, h, D_output=128, base_img_center = None):
    """
    Computes the mask of the inverse planar projection.

    Principle
    ----------

    Principle of the planar projection : Each point of the hemispheric image is projected 
    on a plane orthogonal to the zenith of the observer, and located at a certain altitude 
    H from the position of the observer.

    Parameters
    ----------
    R0 : int
        Radius of the hemispherical image that was initially projected (in pixels)
    h : float
        Ratio between the altitude of the plane and the size of the plane
    D_output : int, optional
        Size of the projection
    base_img_center : (int, int), optional
        Not used

    Returns
    -------
    (coordsproj_i, coordsproj_j)

    coordsproj_i : np.ndarray of shape (D_output, D_output) and dtype int
    coordsproj_j : np.ndarray of shape (D_output, D_output) and dtype int
        (coordsproj_i[k,l], coordsproj_j[k,l]) are the coordinates of the projection of the point (k,l)
    """

    coords_proj = (np.zeros((R0*2,R0*2), dtype = int),np.zeros((R0*2,R0*2), dtype = int))

    for i in range(R0*2):
        for j in range(R0*2):
            if i != R0 or j != R0:
                r = ((i/R0-1)**2 + (j/R0-1)**2)**0.5
                alpha = min(np.pi*r/2, np.pi/2)
                r2 = np.tan(alpha)*h
                cost = (i/R0-1)/r
                sint = (j/R0-1)/r
                i0 = int(max(0,min(D_output-1,cost*r2*D_output/2 + D_output/2)))
                j0 = int(max(0,min(D_output-1,sint*r2*D_output/2 + D_output/2)))
                coords_proj[0][i,j] = i0
                coords_proj[1][i,j] = j0
            else:
                coords_proj[0][i,j] = D_output//2
                coords_proj[1][i,j] = D_output//2           
       
    return coords_proj


def perisolar_mask(time, proj, D, r_mask):

    if time[0:4] == "2020":
        final_df = pd.read_csv('full_df.csv')
    else:
        final_df = pd.read_csv('final_df.csv')

    final_df = final_df.set_index("Time")

    r = 2*np.tan(np.pi/2 - final_df.loc[time,"Altitude"])*0.2
    i_sol, j_sol = int(D//2*r*np.cos(np.pi*final_df.loc[time,"Azimuth"]/180)+D//2), int(D//2*r*np.sin(np.pi*final_df.loc[time,"Azimuth"]/180)+D//2)

    I, J = ogrid[:D, :D]
    dist_from_center = (I - i_sol)**2 + (J - j_sol)**2

    mask = dist_from_center > r_mask**2

    return mask[proj], (i_sol, j_sol)
