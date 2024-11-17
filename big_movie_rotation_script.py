import numpy as np
import re
import os
import json
import time
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def get_entries_in_directory(filename_pattern: str) -> list[os.DirEntry]:
    """
    Gets all matching entries within working directory and retruns a list
    with directory entries as an `os.DirEntry` object and prints the entries.

    Parameters
    ----------
    filename_pattern: str
            General filename_pattern which allows the use of wildcards specified in
            the `re` documentation.
    
    Returns
    -------
    entries_of_interest: list[os.DirEntry]
        A list of directory entries which match the filename_pattern. The items 
        in the list are `os.DirEntry` objects.
    
    Notes
    -----
    Generally, use `.+` to indicate a wildcard in the string. 
    That is, any number and type of characters may occur in place of `.+`.

    Also, make sure you set your desired working directory with a function such
    as `os.chdir(pathtodirectory)` before using the function.

    Usage
    -----
    Suppose your directory contains the following:
    ```
    dataset.xml
    Time_000130_Angle_0_c0_ls_1.ome.tif
    Time_Angle_45_c0_ls_1.ome.tif
    Time_Angle_90_c0_ls_1.ome.tif
    Time_000130_Angle_135_c0_ls_1.ome.tif
    Time_000130_Angle_180_c0_ls_1.ome.tif
    Time_000130_Angle_225_c0_ls_1.ome.tif
    Time_000130_Angle_270_c0_ls_1.ome.tif
    Time_000130_Angle_315_c0_ls_1.ome.tif
    ```
    Then running the function outputs:
    >>> entry_list = get_entries_in_directory('Time_.+_Angle_.+_c0_ls_1.+)
    Entry found: Time_000130_Angle_0_c0_ls_1.ome.tif
    Entry found: Time_000130_Angle_135_c0_ls_1.ome.tif
    Entry found: Time_000130_Angle_180_c0_ls_1.ome.tif
    Entry found: Time_000130_Angle_225_c0_ls_1.ome.tif
    Entry found: Time_000130_Angle_270_c0_ls_1.ome.tif
    Entry found: Time_000130_Angle_315_c0_ls_1.ome.tif
    >>> entry_list
    [<DirEntry 'Time_000130_Angle_0_c0_ls_1.ome.tif'>,
    <DirEntry 'Time_000130_Angle_135_c0_ls_1.ome.tif'>,
    <DirEntry 'Time_000130_Angle_180_c0_ls_1.ome.tif'>,
    <DirEntry 'Time_000130_Angle_225_c0_ls_1.ome.tif'>,
    <DirEntry 'Time_000130_Angle_270_c0_ls_1.ome.tif'>,
    <DirEntry 'Time_000130_Angle_315_c0_ls_1.ome.tif'>]
    """
    iter = os.scandir()
    entries_of_interest = []
    for entry in iter:
        if re.search(filename_pattern, entry.name):
            entries_of_interest.append(entry)
    iter.close()
    for entry in entries_of_interest:
        print(f"Entry found: {entry.name}")
    if entries_of_interest == []:
        print("Warning: No entries found.")
    return entries_of_interest

def get_formatted_entries(filename_template: str, t_ids=None, angles=None) -> list[os.DirEntry]:
    """
    Gets entries similar to how ImageJ does it.
    filename_template example : TP_{t_id}_{angle}_.tif
    """
    entries = []
    if type(t_ids) != type(None) and type(angles) != type(None):
        for t_id in t_ids:
            for angle in angles:
                filename = filename_template.format(t_id=t_id, angle=angle)
                entry = get_entries_in_directory(filename)
                if entry == []:
                    continue
                entries.append(entry)
    elif type(t_ids) != type(None):
        for t_id in t_ids:
            filename = filename_template.format(t_id=t_id)
            entry = get_entries_in_directory(filename)
            if entry == []:
                continue
            entries.append(entry)
    elif type(angles) != type(None):
        for angle in angles:
            filename = filename_template.format(angle=angle)
            entry = get_entries_in_directory(filename)
            if entry == []:
                continue
            entries.append(entry)
    entries = [entry[0] for entry in entries]
    return entries

def ordered_subset(arr1, arr2):
    """Check to see if one array is an ordered subset of another. Arrays are integers"""
    n = len(arr1)
    m = len(arr2)

    if n == m:
        return np.array_equal(arr1, arr2)
    elif n < m:
        for i in range(m-n):
            sub_arr2 = arr2[i:i+n]
            if np.array_equal(arr1, sub_arr2):
                return True
        return False
    elif n > m:
        for i in range(n-m):
            sub_arr1 = arr1[i:i+m]
            if np.array_equal(sub_arr1, arr2):
                return True
        return False
    

if __name__ == "__main__":
    script_start_time = time.time()
    
    """
    Example usage
    >>> json_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    >>> json_name = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8/dataset_8_rotation.json"
    >>> t_ids = np.r_[np.arange(0, 130, 1)]
    >>> img_dir = r"/mnt/crunch/susie/h2afva-gfp/202304281730/Unpacked/"
    >>> img_filename_template = "TP{t_id}_.+.tif"
    >>> rotated_img_save_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8/dataset_8_rotated_big_tiff"
    >>> downsampling = 0.25
    """
    # script parameters
    # rotation .json parameters
    json_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    json_name = r"dataset_8_rotation.json"
    # img parameters
    t_ids = np.r_[np.arange(0, 130, 1)]
    img_dir = r"/mnt/crunch/susie/h2afva-gfp/202307101735/Time4views_1p5mum_1p0msexposure/data"
    img_filename_template = "TP{t_id}_.+.tif"
    # rotated tiff parameters
    rotated_img_save_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8/dataset_8_rotated_big_tiff"
    # downsampling factor which was used to calculate rotations 
    downsampling = 0.25

    # Display parameters for confirmation
    print("Please confirm the parameters:")
    print(f"json_dir: {json_dir}")
    print(f"json_name: {json_name}")
    print(f"t_ids: {t_ids}")
    print(f"img_dir: {img_dir}")
    print(f"img_filename_template: {img_filename_template}")
    print(f"rotated_img_save_dir: {rotated_img_save_dir}")
    print(f"downsampling: {downsampling}")

    # Confirmation prompt
    confirmation = input("Are these parameters correct? Type 'yes' to proceed or 'no' to exit: ").strip().lower()

    if confirmation != 'yes':
        print("Execution stopped by user.")
        exit()
    else:
        print("Parameters confirmed. Continuing execution...")

    # find all images
    os.chdir(img_dir)
    entries = get_formatted_entries(img_filename_template, t_ids=t_ids)
    entries_path = [os.path.join(img_dir, entry.name) for entry in entries]
    entry_names = [entry.name for entry in entries]

    # read in stored json 
    with open(os.path.join(json_dir, json_name), 'r') as f:
        dataset = json.load(f)

    # sort by time if not already
    dataset = dict(sorted(dataset.items(), key=lambda x: int(x[0][4:])))

    # convert back necessary values back to nd_array
    for d in dataset.values():
        d['center'] = np.array(d['center'])
        d['rotation_to_next'] = np.array(d['rotation_to_next'])

    # assert t_ids order matches the time order of .json
    t_ids_from_keys = np.array([int(x[4:]) for x in dataset.keys()])

    if ordered_subset(t_ids, t_ids_from_keys):
        pass
    else:
        print("Warning, t_ids and time tokens are not aligned. Exiting...")
        exit()
    
    # try to make the save directory
    try:
        os.mkdir(rotated_img_save_dir)
        print(f"Made direcotry {rotated_img_save_dir}.")
    except:
        print(f"The directory {rotated_img_save_dir} already exists.")
        pass

    # main loop
    # registers with the first time point as fixed.
    for i, t in enumerate(t_ids):
        loop_start_time = time.time()

        # first frame is fixed
        if t == t_ids[0]:
            continue

        token = f"time{t}"
        center = dataset[token]['center']
        padding = dataset[token]['padding']

        # this is the rotation we apply to timepoint n
        if t == t_ids[1]:
            rotation_prev = np.eye(3)
        else:
            rotation_prev = rotation
        rotation = rotation_prev @ dataset[f"time{t_ids[i-1]}"]['rotation_to_next'].T

        # read image 
        print(f"Reading image {entry_names[i]}")
        im = io.imread(entries_path[i])
        
        # pad image
        padding_rs = int(np.round(padding/downsampling))
        im_padded = np.pad(im, pad_width=padding_rs, mode='constant', constant_values=0)

        # calculate new center
        center_rs = center/downsampling

        # transform image
        print(f"Affine transforming...")
        affine_matrix = (tcreg.package_affine_transformation(np.eye(3), center_rs) @
                    tcreg.package_affine_transformation(rotation.T, np.zeros(3)) @
                    tcreg.package_affine_transformation(np.eye(3), -center_rs))   
        im_rotated = ndimage.affine_transform(im_padded, affine_matrix, output_shape=None, order=1,
                                        mode='constant', cval=0.0, prefilter=False) # bilinear interpolation
        
        # save image
        print(f"Saving image...")
        tifffile.imwrite(os.path.join(rotated_img_save_dir, entry_names[i]), im_rotated.astype("uint16"), imagej=True, metadata={"axes": "ZYX"})

        loop_end_time = time.time()
        hrs, min, s = format_elapsed_time(loop_end_time - loop_start_time)
        print(f"Time per loop: {int(hrs)}:{int(min)}:{s:.6f}" )
    
    script_end_time = time.time()
    hrs, min, s = format_elapsed_time(script_end_time - script_start_time)
    print(f"Elapsed time: {int(hrs)}:{int(min)}:{s:.6f}")
    print("SCRIPT COMPLETE")