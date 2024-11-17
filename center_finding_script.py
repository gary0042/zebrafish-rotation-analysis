from spherical_analysis import *
from scipy.optimize import minimize
import time
import json 
import os
import re
import sys

def format_elapsed_time(time):
    """
    Returns the number of seconds elapsed and formats it into hours, minutes, and seconds.
    """
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    return (hours, minutes, seconds)

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

if __name__ == "__main__":
    script_start_time = time.time()

    # script parameters
    """
    Example
    >>> data_dir = r"/mnt/data1/Code/GH_local/DS5_202304281730_downsampled_0point3x/"
    >>> filename_template = "DS_TP{t_id}_.+.tif"
    >>> t_ids = np.arange(35, 40, 1)
    >>> resolution = [0.831, 0.831, 0.831] 
    >>> padding = 60 # requires tweaking
    >>> alpha = 0.1 # requires tweaking
    >>> json_save_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_8"
    >>> json_save_name = "dataset_5_test.json"
    >>> seed_time = 35
    >>> thickness = 16
    """
    data_dir = r"/mnt/data1/Code/GH_local/DS5_202304281730_downsampled_0point3x/"
    filename_template = "DS_TP{t_id}_.+.tif"
    t_ids = np.r_[np.arange(0, 32, 1), np.arange(33, 130, 1)]
    resolution = [0.831, 0.831, 0.831] 
    padding = 60 # requires tweaking
    alpha = 0.1 # requires tweaking
    json_save_dir = r"/mnt/data1/Code/GH_local/spherical_harmonic_rotation_analysis/dataset_5"
    json_save_name = "dataset_5_new.json"
    seed_time = 35
    thickness = 64 # number of pixels in the radial direction to integrate over. Should be large enough to encapsulate entire fish.

    # Display parameters for confirmation
    print("Please confirm the parameters:")
    print(f"data_dir: {data_dir}")
    print(f"filename_template: {filename_template}")
    print(f"t_ids: {t_ids}")
    print(f"resolution: {resolution}")
    print(f"padding: {padding}")
    print(f"alpha: {alpha}")
    print(f"json_save_dir: {json_save_dir}")
    print(f"json_save_name: {json_save_name}")
    print(f"seed_time: {seed_time}")
    print(f"thickness: {thickness}")

    # Confirmation prompt
    confirmation = input("Are these parameters correct? Type 'yes' to proceed or 'no' to exit: ").strip().lower()

    if confirmation != 'yes':
        print("Execution stopped by user.")
        exit()
    else:
        print("Parameters confirmed. Continuing execution...")

    # check seed_time is in t_ids
    try:
        seed_time_index = np.where(t_ids==seed_time)[0][0]
    except:
        print("Seed time not in t_ids. Exiting...")
        sys.exit(0) 
        

    # Find all entries
    os.chdir(data_dir)
    entires = get_formatted_entries(filename_template, t_ids=t_ids)
    entries = [os.path.join(data_dir, entry.name) for entry in entires]

    # create dataset
    dataset = {}
    for i, entry in enumerate(entries):
        t = t_ids[i]
        dataset[f'time{t}'] = {}
        dataset[f'time{t}']['resolution'] = resolution
        dataset[f'time{t}']['padding'] = padding
        dataset[f'time{t}']['alpha'] = alpha
        dataset[f'time{t}']['path'] = entry
    
    # sort dataset by time
    dataset = dict(sorted(dataset.items(), key=lambda x: int(x[0][4:])))

    # save initial .json
    print("Saving .json...")
    save_json(dataset, json_save_dir, json_save_name)

    # run full script on the seed time
    seed_start_time = time.time()
    token = f"time{seed_time}"
    d = dataset[token]
    print("Reading image for time ", seed_time)
    img = io.imread(d['path'])
    img_processed = clip_pad_image(img, alpha, padding, visual=False)
    img_interp = create_interpolated_volume(img_processed)
    overlap = create_overlap_function(img_interp, thickness=8, r_res = 64, theta_res = 128, phi_res=128)
    x_0 = find_radius_brute_force(overlap, img_processed, center_range=50, radius_range=50, center_steps=3, radius_steps=3)
    param_refined = minimize(overlap, x_0, method='Nelder-Mead', options={'maxfev': 50, 'disp': False})
    d['center'] = param_refined.x[0:-1]
    d['radius'] = param_refined.x[-1]
    seed_end_time = time.time()
    hrs, min, s = format_elapsed_time(seed_end_time - seed_start_time)
    print(f"Time per initial loop: {int(hrs)}:{int(min)}:{s:.6f}")

    # propogate 
    seed_time_index = np.where(t_ids==seed_time)[0][0]

    # make the lists for backward and forward time propogation starting from seed_time
    backward_t = []
    for t in t_ids[seed_time_index::-1]:
        backward_t.append(t)
    forward_t = []
    for t in t_ids[seed_time_index::1]:
        forward_t.append(t)
    
    # remove the seed time from both lists
    del backward_t[0]
    del forward_t[0]

    # set previous radius and previous center
    prev_radius = dataset[f'time{seed_time}']['radius']
    prev_center = dataset[f'time{seed_time}']['center']

    # backward
    for t in backward_t:
        loop_start_time = time.time()

        # set the token that will be used to access the dictionary for that time
        token = f"time{t}"

        # read in image for that time
        print("Reading image for time ", t)
        img = io.imread(dataset[token]['path'])
        
        # preprocess
        img_processed = clip_pad_image(img, alpha, padding, visual=False)

        # two step optimization using previouos parameters
        img_interp = create_interpolated_volume(img_processed)

        # step 1 is to optimize center, keeping radius constant
        overlap1 = create_overlap_function(img_interp, radius=prev_radius, thickness=8, r_res = 64, theta_res = 128, phi_res=128)
        res_cent = minimize(overlap1, prev_center, method='Nelder-Mead', options={'maxfev': 50, 'disp': False})

        # step 2 is to optimize radius, keeping center constant
        overlap2 = create_overlap_function(img_interp, center=prev_center, thickness=8, r_res = 64, theta_res = 128, phi_res=128)
        res_rad = minimize(overlap2, prev_radius, method='Nelder-Mead', options={'maxfev': 50, 'disp': False})

        # store results
        dataset[token]['radius'] = res_rad.x[0]
        dataset[token]['center'] = res_cent.x

        # set the previous center and radius to the one you just computed
        prev_center = res_cent.x
        prev_rad = res_rad.x

        # save every so often
        if t % 5 == 0:
            print("Saving .json...")
            save_json(dataset, json_save_dir, json_save_name)
        
        loop_end_time = time.time()
        hrs, min, s = format_elapsed_time(loop_end_time - loop_start_time)
        print(f"Time per loop: {int(hrs)}:{int(min)}:{s:.6f}" )

    save_json(dataset, json_save_dir, json_save_name)

    for t in forward_t:
        loop_start_time = time.time()

        # set the token that will be used to access the dictionary for that time
        token = f"time{t}"

        # read in image for that time
        print("Reading image for time ", t)
        img = io.imread(dataset[token]['path'])
        
        # preprocess
        img_processed = clip_pad_image(img, alpha, padding, visual=False)

        # two step optimization using previouos parameters
        img_interp = create_interpolated_volume(img_processed)

        # step 1 is to optimize center, keeping radius constant
        overlap1 = create_overlap_function(img_interp, radius=prev_radius, thickness=8, r_res = 64, theta_res = 128, phi_res=128)
        res_cent = minimize(overlap1, prev_center, method='Nelder-Mead', options={'maxfev': 50, 'disp': False})

        # step 2 is to optimize radius, keeping center constant
        overlap2 = create_overlap_function(img_interp, center=prev_center, thickness=8, r_res = 64, theta_res = 128, phi_res=128)
        res_rad = minimize(overlap2, prev_radius, method='Nelder-Mead', options={'maxfev': 50, 'disp': False})

        # store results
        dataset[token]['radius'] = res_rad.x[0]
        dataset[token]['center'] = res_cent.x

        # set the previous center and radius to the one you just computed
        prev_center = res_cent.x
        prev_rad = res_rad.x

        # save every so often
        if t % 5 == 0:
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