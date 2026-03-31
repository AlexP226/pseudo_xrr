# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:25:50 2026

@author: shenc
"""
import numpy as np
import os
import numbers


def check_keys_numeric(keys, *dicts):
    """
    new function from Chen
    keys  – iterable of required keys (e.g., ['flux','cttime'])
    dicts – any number of dictionaries to check
    """
    for key in keys:
        for d in dicts:
            if key not in d:
                return False
            if not isinstance(d[key], numbers.Number):
                return False
    return True    

def mean1d_if_within_percent(arr, tol=0.01):
    """
    If all values differ from the mean by less than tol (relative),
    replace by the mean while preserving a compatible shape:
      (1,n) -> (1,1)
      (n,)  -> (1,)
    Otherwise return the original array.
    """
    a = np.asarray(arr, dtype=float)
    flat = a.ravel()
    m = flat.mean()

    scale = max(abs(m), 1e-12)
    max_rel_dev = np.max(np.abs(flat - m)) / scale

    if max_rel_dev < tol:
        print("array values vary < tolerance %.2f%%, average" % (tol * 100))
        if a.ndim == 2 and a.shape[0] == 1:
            return np.array([[m]], dtype=float)   # (1,1)
        elif a.ndim == 1:
            return np.array([m], dtype=float)     # (1,)
        else:
            # fallback for unexpected shapes
            return np.full_like(a, m)
    else:
        print("array values vary > tolerance %.2f%%, keep as array" % (tol * 100))
        return a
    
    

def make_filename(metadata, suffix=None):
    """
    Build a general output filename for data or plots from metadata.

    Naming:
    - single scan -> <sample>_<id>_<suffix>
    - scan array  -> <sample>_<id_min>_<id_max>_<suffix>

    where each id is formatted as at least 5 digits with leading zeros.

    Examples
    --------
    suffix = "SF.dat"    -> sample_00123_SF.dat
    suffix = "R.png"     -> sample_00123_R.png
    suffix = "Qxy.png"   -> sample_00123_00130_Qxy.png

    Special case
    ------------
    If metadata["facility"] indicates PETRA III/P08 and the sample name starts
    with "GID_", that prefix is removed from the filename.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing at least:
        - metadata["paths"]["path_out"]
        - metadata["measurements"]["sample"]
        - metadata["measurements"]["scan"]

    suffix : str
        Required output suffix, e.g. "SF.dat", "R.dat", "R.png", "Qxy.png".

    Returns
    -------
    filename : str
        Full output path including directory and filename.
    """
    if metadata is None:
        raise ValueError("metadata is required.")
    if suffix is None:
        raise ValueError("suffix must be provided.")

    if "paths" not in metadata or metadata["paths"] is None:
        raise ValueError("metadata['paths'] is required.")
    if "measurements" not in metadata or metadata["measurements"] is None:
        raise ValueError("metadata['measurements'] is required.")

    path_out = metadata["paths"].get("path_out", None)
    sample = metadata["measurements"].get("sample", None)
    scan = metadata["measurements"].get("scan", None)
    facility = metadata.get("facility", None)

    if path_out is None:
        raise ValueError("metadata['paths']['path_out'] is required.")
    if sample is None:
        raise ValueError("metadata['measurements']['sample'] is required.")
    if scan is None:
        raise ValueError("metadata['measurements']['scan'] is required.")

    # facility-specific sample-name cleanup
    if facility is not None:
        facility_str = str(facility)
        if ("PETRA III" in facility_str) and ("P08" in facility_str):
            if str(sample).startswith("GID_"):
                sample = str(sample)[4:]

    scan_arr = np.asarray(scan).ravel()
    if scan_arr.size == 0:
        raise ValueError("metadata['measurements']['scan'] must not be empty.")

    if scan_arr.size == 1:
        scan_part = f"{int(scan_arr[0]):05d}"
    else:
        scan_part = f"{int(np.min(scan_arr)):05d}_{int(np.max(scan_arr)):05d}"

    filename = f"{sample}_{scan_part}_{suffix}"
    return os.path.join(path_out, filename)
