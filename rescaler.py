"""
rescaler.py

A script for rescaling images.
"""

from __future__ import annotations
import tifffile
import os
import re  # used for wildcards
import numpy as np
from skimage import io, transform, util
import time


def format_elapsed_time(time):
    """
    Returns the number of seconds elapsed and formats it into hours, minutes, and seconds.
    """
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    return (hours, minutes, seconds)


def get_entries_in_directory(filename_pattern: str) -> list[os.DirEntry]:
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


def get_formatted_entries(
    filename_template: str, t_ids=None, angles=None
) -> list[os.DirEntry]:
    """filename_template example : TP_{t_id}_{angle}_.tif"""
    entries = []
    if t_ids != None and angles != None:
        for t_id in t_ids:
            for angle in angles:
                filename = filename_template.format(t_id=t_id, angle=angle)
                entry = get_entries_in_directory(filename)
                if entry == []:
                    continue
                entries.append(entry)
    if t_ids != None:
        for t_id in t_ids:
            filename = filename_template.format(t_id=t_id)
            entry = get_entries_in_directory(filename)
            if entry == []:
                continue
            entries.append(entry)
    entries = [entry[0] for entry in entries]
    return entries


if __name__ == "__main__":
    # ===========================================================================================
    # PARAMETERS
    # Set working directory
    datadir = r"/mnt/crunch/susie/h2afva-gfp/202307101735/Time4views_1p5mum_1p0msexposure/data/"

    # Set directory where you want to save the rescaled stacks
    savedir = r"/mnt/data1/Code/GH_local/rescaler/DS8_20230710735_downsampled_0point25x"

    # Set prefix of rescaled stacks
    saveprefix = r"DS_"

    # Set number of timepoints (Must be a list, not numpy array)
    times = np.r_[np.arange(0, 151, 1)]

    # Filename template
    template = r"TP{t_id}_.+.tif"

    # scaling
    rescaling_factor = (0.25, 0.25, 0.25)
    # ===========================================================================================
    # MAIN SCRIPT
    start_time = time.time()
    # Detect entries
    os.chdir(datadir)
    entries = get_formatted_entries(template, t_ids=times)

    # Rescale and save
    for entry in entries:
        print("Processing: ", entry.name)
        stk = tifffile.imread(entry.name)
        stk_ds = transform.rescale(stk, rescaling_factor, preserve_range=True)
        print("Rescaled size: ", stk_ds.shape)

        savename = saveprefix + entry.name
        savefile = os.path.join(savedir, savename)
        tifffile.imwrite(savefile, stk_ds.astype("uint16"), imagej=True, metadata={"axes": "ZYX"})

    # DONE
    end_time = time.time()
    hrs, min, s = format_elapsed_time(end_time - start_time)
    print(f"Elapsed time: {int(hrs)}:{int(min)}:{s:.6f}")
    print("SCRIPT COMPLETE")
