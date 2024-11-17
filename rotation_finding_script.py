import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import jit
from skimage import io, exposure
from skimage.filters import threshold_otsu
from scipy import interpolate as interp
from scipy import ndimage, stats, special
from scipy.optimize import minimize, brute
from blender_tissue_cartography import rotation as tcrot
from blender_tissue_cartography import registration as tcreg
from blender_tissue_cartography import harmonic as tcharm
from spherical_analysis import * 

def save_json(dataset, save_dir, save_name):
    def turn_numpy_into_list(d):
        for key, val in d.items():
            if type(val) == type(np.array([0])):
                d[key] = val.tolist()
    
    # turn numpy arrays into lists (2 levels deep)
    turn_numpy_into_list(dataset)
    for key, val in dataset.items():
        if type(val) == type({}):
            turn_numpy_into_list(val)
    
    with open(os.path.join(save_dir, save_name), 'w') as f:
        json.dump(dataset, f, indent=4)

def format_elapsed_time(time):
    """
    Returns the number of seconds elapsed and formats it into hours, minutes, and seconds.
    """
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    return (hours, minutes, seconds)


if __name__ == "__main__":
    script_start_time = time.time()

    # script parameters
    """
    Example usage
    >>> json_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    >>> json_name = r"dataset_8.json"
    >>> json_save_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    >>> json_save_name = r"dataset_8_rotation.json"
    >>> thickness = 16
    >>> uv_grid_steps = 128
    >>> ell_max = 6
    """
    # .json should have centers and radii
    json_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    json_name = r"dataset_8.json"
    # .json save parameters
    json_save_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    json_save_name = r"dataset_8_rotation.json"
    # rotation detection parameters
    thickness = 32 # number of pixels in the radial direction to integrate over. Should be large enough to encapsulate entire fish.
    uv_grid_steps = 128 # number of grid steps to use when integrating
    ell_max = 10 # max l to compute up to for spherical harmonic coefficients
    
    # Display parameters for confirmation
    print("Please confirm the parameters:")
    print(f"json_dir: {json_dir}")
    print(f"json_name: {json_name}")
    print(f"json_save_dir: {json_save_dir}")
    print(f"json_save_name: {json_save_name}")
    print(f"thickness: {thickness}")
    print(f"uv_grid_steps: {uv_grid_steps}")
    print(f"ell_max: {ell_max}")

    # Confirmation prompt
    confirmation = input("Are these parameters correct? Type 'yes' to proceed or 'no' to exit: ").strip().lower()

    if confirmation != 'yes':
        print("Execution stopped by user.")
        exit()
    else:
        print("Parameters confirmed. Continuing execution...")

    # read in stored json 
    with open(os.path.join(json_dir, json_name), 'r') as f:
        dataset = json.load(f)

    # sort by time if not already
    dataset = dict(sorted(dataset.items(), key=lambda x: int(x[0][4:])))

    # convert back necessary values back to nd_array
    for d in dataset.values():
        d['center'] = np.array(d['center'])

    # create fields for rotation 
    for t, d in dataset.items():
        if d.get('rotation_to_next'):
            continue
        if d.get('quaternion_to_next'):
            continue
        d['rotation_to_next'] = 0
        d['quaternion_to_next'] = 0
    
    # integration grid
    phi_grid, theta_grid = np.meshgrid(np.arange(uv_grid_steps), np.arange(uv_grid_steps))
    phi_grid = 2*np.pi*phi_grid/uv_grid_steps
    theta_grid = np.pi*theta_grid/uv_grid_steps

    # create measure
    dTheta, dPhi = (np.pi/uv_grid_steps, 2*np.pi/uv_grid_steps)
    weights = np.sin(theta_grid)*dTheta*dPhi # spherical integration measure

    token_list = list(dataset.keys())

    # finds maximum time in the entire dataset
    token_max = list(dataset.keys())[-1]

    # begin loop
    for i, token in enumerate(token_list):
        loop_start_time = time.time()

        if token == token_max:
            continue
        d_A = dataset[token_list[i]]
        d_B = dataset[token_list[i+1]]
        print(f"Computing rotation from {token_list[i]} to {token_list[i+1]}")

        # preprocess & pullbacks & spherical harmonic coefficients
        if i == 0:
            img_A = io.imread(d_A['path'])
            img_A_pp = clip_pad_image(img_A, d_A['alpha'], d_A['padding'], visual=False)
            img_A_interp = create_interpolated_volume(img_A_pp)
            A_pb = np.sum(spherical_pullback(img_A_interp, d_A['center'], d_A['radius'], thickness=thickness, r_res=64, theta_res=128, phi_res=128), axis=0)
            A_pb = (A_pb - np.mean(A_pb))/np.std(A_pb) # normalize 
            coeffs_A = tcrot.compute_spherical_harmonics_coeffs(A_pb, phi_grid, theta_grid, weights=weights, max_l=ell_max)
        elif i > 0:
            coeffs_A = coeffs_B
        
        img_B = io.imread(d_B['path'])
        img_B_pp = clip_pad_image(img_B, d_B['alpha'], d_B['padding'], visual=False)
        img_B_interp = create_interpolated_volume(img_B_pp)
        B_pb = np.sum(spherical_pullback(img_B_interp, d_B['center'], d_B['radius'], thickness=thickness, r_res=64, theta_res=128, phi_res=128), axis=0)
        B_pb = (B_pb - np.mean(B_pb))/np.std(B_pb) # normalize
        coeffs_B = tcrot.compute_spherical_harmonics_coeffs(B_pb, phi_grid, theta_grid, weights=weights, max_l=ell_max)

        # get rotation
        rotation_inferred, overlap = tcrot.rotational_alignment(coeffs_A, coeffs_B, allow_flip=False, n_subdiv_axes=2, n_angle=200, maxfev=500)
        q_inferred = tcrot.rot_mat_to_quaternion(rotation_inferred)
        
        # write to dataset metadata
        dataset[token]['rotation_to_next'] = rotation_inferred
        dataset[token]['quaternion_to_next'] = q_inferred

        # save every so often
        if i % 5 == 0:
            print("Saving .json...")
            save_json(dataset, json_save_dir, json_save_name)
        
        loop_end_time = time.time()
        hrs, min, s = format_elapsed_time(loop_end_time - loop_start_time)
        print(f"Time per loop: {int(hrs)}:{int(min)}:{s:.6f}" )

    print("Saving .json...")
    save_json(dataset, json_save_dir, json_save_name)

    script_end_time = time.time()
    hrs, min, s = format_elapsed_time(script_end_time - script_start_time)
    print(f"Elapsed time: {int(hrs)}:{int(min)}:{s:.6f}")
    print("SCRIPT COMPLETE")