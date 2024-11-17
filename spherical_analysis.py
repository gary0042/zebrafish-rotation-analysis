import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, exposure
from skimage.filters import threshold_otsu
from scipy import interpolate as interp
from scipy import ndimage, stats, special

def create_r_theta_phi_interpolation(cartesian_interpolator, sphereical_to_cartesian):
    
    def r_theta_phi_interpolator(spherical_point):
        x, y, z = sphereical_to_cartesian(spherical_point[0], spherical_point[1], spherical_point[2]) 
        return cartesian_interpolator((x, y, z))
    
    return r_theta_phi_interpolator

def create_spherical_coords_given_center(center_x, center_y, center_z):

    # TODO: code xyz to sphereical, will hold off on this as it isn't required in this workflow
    # def convert_from_xy_to_polar(x, y):
    #     # shift coordinates
    #     x_shifted = x - center_x
    #     y_shifted = y - center_y
    #     # calculate ceneter
    #     r = np.sqrt(x_shifted**2 + y_shifted**2)
    #     phi = np.arctan2(y_shifted, x_shifted)
    #     return r, phi
    
    def convert_from_spherical_to_xyz(r, theta, phi):
        return r*np.sin(theta)*np.cos(phi) + center_x, r*np.sin(theta)*np.sin(phi) + center_y, r*np.cos(theta) + center_z
    
    return convert_from_spherical_to_xyz

def create_interpolated_volume(vol):
    x_max, y_max, z_max = vol.shape
    x_pts = np.arange(0, x_max, 1)
    y_pts = np.arange(0, y_max, 1)
    z_pts = np.arange(0, z_max, 1)
    x_grid, y_grid, z_grid = np.meshgrid(x_pts, y_pts, z_pts, indexing='ij')
    return interp.RegularGridInterpolator((x_pts, y_pts, z_pts), vol[x_grid, y_grid, z_grid], bounds_error=False, fill_value=0)

def spherical_pullback(f_vol_cartesian, center, radius, thickness=8, r_res = 64, theta_res = 128, phi_res=128):
    # make the maps (x, y, z) -> (r, theta, phi) and
    # g(x(r, theta, phi), y(r, theta, phi), z(r, theta, phi))
    spherical_to_xyz = create_spherical_coords_given_center(*center)
    f_vol_spherical = create_r_theta_phi_interpolation(f_vol_cartesian, spherical_to_xyz)

    # sample points from g(.)
    r_pts = np.linspace(radius-thickness/2, radius+thickness/2, r_res)
    theta_pts = np.linspace(0, np.pi, theta_res)
    phi_pts = np.linspace(0, 2*np.pi, phi_res)
    r_grid, theta_grid, phi_grid  = np.meshgrid(r_pts, theta_pts, phi_pts, indexing='ij')
    sample = f_vol_spherical((r_grid, theta_grid, phi_grid))
    return sample

def create_r_theta_phi_interpolation(cartesian_interpolator, sphereical_to_cartesian):
    
    def r_theta_phi_interpolator(spherical_point):
        x, y, z = sphereical_to_cartesian(spherical_point[0], spherical_point[1], spherical_point[2]) 
        return cartesian_interpolator((x, y, z))
    
    return r_theta_phi_interpolator

def create_spherical_coords_given_center(center_x, center_y, center_z):

    # TODO: code xyz to sphereical, will hold off on this as it isn't required in this workflow
    # def convert_from_xy_to_polar(x, y):
    #     # shift coordinates
    #     x_shifted = x - center_x
    #     y_shifted = y - center_y
    #     # calculate ceneter
    #     r = np.sqrt(x_shifted**2 + y_shifted**2)
    #     phi = np.arctan2(y_shifted, x_shifted)
    #     return r, phi
    
    def convert_from_spherical_to_xyz(r, theta, phi):
        return r*np.sin(theta)*np.cos(phi) + center_x, r*np.sin(theta)*np.sin(phi) + center_y, r*np.cos(theta) + center_z
    
    return convert_from_spherical_to_xyz

def create_interpolated_volume(vol): 
    x_max, y_max, z_max = vol.shape
    x_pts = np.arange(0, x_max, 1)
    y_pts = np.arange(0, y_max, 1)
    z_pts = np.arange(0, z_max, 1)
    # x_grid, y_grid, z_grid = np.meshgrid(x_pts, y_pts, z_pts, indexing='ij') # irrelevant 
    return interp.RegularGridInterpolator((x_pts, y_pts, z_pts), vol, bounds_error=False, fill_value=0)

def spherical_pullback(f_vol_cartesian, center, radius, thickness=8, r_res = 64, theta_res = 128, phi_res=128):
    # make the maps (x, y, z) -> (r, theta, phi) and
    # g(x(r, theta, phi), y(r, theta, phi), z(r, theta, phi))
    # print(center)
    spherical_to_xyz = create_spherical_coords_given_center(*center)
    f_vol_spherical = create_r_theta_phi_interpolation(f_vol_cartesian, spherical_to_xyz)

    # sample points from g(.)
    r_pts = np.linspace(radius-thickness/2, radius+thickness/2, r_res)
    theta_pts = np.linspace(0, np.pi, theta_res)
    phi_pts = np.linspace(0, 2*np.pi, phi_res)
    r_grid, theta_grid, phi_grid  = np.meshgrid(r_pts, theta_pts, phi_pts, indexing='ij')
    sample = f_vol_spherical((r_grid, theta_grid, phi_grid))
    return sample

def create_rotated_volume(cartesian_interpolator, R, center):
    """Samples the cartesian interpolated at rotated points. Basically a precomposition."""
    def rotated_cartesian_interpolator(cartesian_point):
        # assume some grid form
        x, y, z = (cartesian_point[0]-center[0], cartesian_point[1]-center[1], cartesian_point[2]-center[2])
        coordinates = np.vstack([x.ravel(), y.ravel(), z.ravel()]) 
        coordinates_rot = R @ coordinates
        x_rot, y_rot, z_rot = (coordinates_rot[0].reshape(x.shape), coordinates_rot[1].reshape(y.shape), coordinates_rot[2].reshape(z.shape)) 
        return cartesian_interpolator((x_rot+center[0], y_rot+center[1], z_rot+center[2]))
    
    return rotated_cartesian_interpolator

def visualize_center(image, center, r):
    """View an volumetric image with its circumcircle."""
    side1 = image[int(center[0]), ...]
    side2 = image[..., int(center[-1])]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    ax1.imshow(side1)
    ax2.imshow(side2)
    ax1.plot(center[2], center[1], marker='o', c='red') 
    ax2.plot(center[1], center[0], marker='o', c='red')
    # circle parameters
    circle1_center = (center[2], center[1])
    circle2_center = (center[1], center[0])
    circle1 = patches.Circle(circle1_center, r, edgecolor='red', facecolor='none', linewidth=1)
    circle2 = patches.Circle(circle2_center, r, edgecolor='red', facecolor='none', linewidth=1)
    # add circles to axes
    ax1.add_patch(circle1)
    ax2.add_patch(circle2)

def find_radius_brute_force(overlap_function, image, center_range=50, radius_range=50, center_steps=3, radius_steps=3):
    """Does a brute force search over a parameter space. Number of samples = center_steps^3 * radius_steps."""
    c_0 = np.array(image.shape)/2
    r_0 = min(image.shape)/2
    
    # center trial points are a cube about the image center
    c_x_pts = np.linspace(c_0[0]-center_range/2, c_0[0]+center_range/2, num=center_steps)
    c_y_pts = np.linspace(c_0[1]-center_range/2, c_0[1]+center_range/2, num=center_steps)
    c_z_pts = np.linspace(c_0[2]-center_range/2, c_0[2]+center_range/2, num=center_steps)

    # radius trial points are values that go inwards from the minimum of the image shape/2
    r_pts = np.linspace(r_0, r_0-radius_range, radius_steps)

    # Make a list of all parameter combinations 
    parameter_grid = np.meshgrid(c_x_pts, c_y_pts, c_z_pts, r_pts)
    parameter_list = np.vstack([np.ravel(x) for x in parameter_grid]).T

    # brute force. keep a running minimum
    min_val = 0
    min_param = None
    for parameter in parameter_list:
        res = overlap_function(parameter)
        if res < min_val:
            min_val = res
            min_param = parameter
            
    return min_param

def create_overlap_function(f_vol_cartesian, center=None, radius=None, **kwargs):

    if type(center)==type(None) and radius==None:
        def overlap_function(X, **kwargs):
            # X = (center_x, center_y, center_z, radius)
            pb = spherical_pullback(f_vol_cartesian, X[0:-1], X[-1], **kwargs)
            overlap = np.sum(pb)
            return -overlap # return negative since we want it to be minimized
        
        return overlap_function
    
    elif radius!=None and type(center)==type(None):
        def overlap_function_const_radius(X, **kwargs):
            # X = (center_x, center_y, center_z)
            pb = spherical_pullback(f_vol_cartesian, X, radius, **kwargs)
            overlap = np.sum(pb)
            return -overlap # return negative since we want it to be minimized
        
        return overlap_function_const_radius
    
    elif type(center)!=type(None) and radius==None:
        def overlap_function_const_center(X, **kwargs):
            # X = (radius)
            pb = spherical_pullback(f_vol_cartesian, center, X, **kwargs)
            overlap = np.sum(pb)
            return -overlap # return negative since we want it to be minimized
        
        return overlap_function_const_center

def clip_pad_image(image, alpha, padding, visual=False):
    """Clips image at the alpha_th quantile. In addition, pads the image Can visualize if necessary."""
    q = [alpha, 1 - alpha]
    mquantiles = stats.mstats.mquantiles(image.flatten(), prob=q)
    img_clip = np.clip(image - mquantiles[0], a_min=0, a_max=np.inf)
    img_final = np.pad(img_clip, pad_width=padding, mode='constant', constant_values=0)

    if visual:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(24, 6))
        ax1.imshow(img_final[img_final.shape[0]//2, ...])
        hist_c, hist_c_centers = exposure.histogram(img_clip)
        hist, hist_centers = exposure.histogram(image) 
        ax2.plot(hist_c_centers, hist_c)
        ax3.plot(hist_centers, hist)
        ax1.set_title("Clipped image")   
        ax2.set_title("Clipped histogram")
        ax3.set_title("Original histogram")    

    return img_final



