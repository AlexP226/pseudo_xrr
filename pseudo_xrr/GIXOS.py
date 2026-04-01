import numpy as np
import matplotlib.pyplot as plt

import math
#import os
#import platform
#from joblib import Parallel, delayed
#from scipy.integrate import dblquad
#from scipy.special import kv as besselk, jv as besselj, gamma
from p08_GIXD.p08_GIXD import *
from pseudo_xrr.helpers import *
from pseudo_xrr.eCWM import *
from pseudo_xrr.bulkbkg import *
from pseudo_xrr.plot import *

'''
change oct.2025
author shenc
(1) load_metadata 
- use YAML from ruamel.yaml
    - 1e11 is loaded as str while 1e+11 and 1e-11 are loaded as number in yaml 1.1 that is used by PyYAML
    - ruamel.yaml uses yaml 1.2 that can cope with the three automatically
- add datatype: 1d or 2d
- calculation of HW in the load_metadata already
(2) import_data changed into import_data_from_meta: 
- this is for loading data from metadata info.
- add loading 2d data, with tt and tth axises
(3) making an import_data jsut for importing 2D and 1D data
(4) change DS/RRF into roughness factor form
(5) change the integral of angle from degree to radian and remove the *(pi/180)^2 from the scaling accordingly


'''



        

#%% data processing and correction
def geometrical_corr(gixs2d, Ddet = 560.7, det_px = 0.075, HWtth = None, HWtt = None):
    """
    geometrical correction of 2d data, 
    - to be used in case the geometrical correction is not done in the rebinned data
    - for detector perpendicular to the surface
    - It takes the detector px size (mm) and angular HW of the cell (deg) to calculate the correction
    
    Parameters
    ----------
    gixs2d : dictionary
        at least with Intensity, tt, tth.
    Ddet : float, optional
        [mm] detector sample distance. The default is 560.7.
    det_px : float, optional
        [mm] pixel size. The default is 0.075.
    HWtth : float, optional
        [deg] half width in tth. The default is None.
    HWtt : float, optional
        [deg] half width in tt. The default is None.

    Returns
    -------
    gixs2d : dictionary
        field:
            'Intensity':    corrected intensity
            'error':        sqrt of the intensity after correction
            preserve all the other fields

    """
    correction_tt = np.ones((len(gixs2d["tt"]),))
    correction_tth = np.ones((1,gixs2d["tth"].shape[1]))
    if HWtt is not None and isinstance(HWtt, numbers.Number):
        correction_tt = np.radians(HWtt*2) / (np.arctan((np.tan(np.radians(gixs2d["tt"]))*Ddet + det_px/2)/Ddet) - np.arctan((np.tan(np.radians(gixs2d["tt"]))*Ddet - det_px/2)/Ddet))
    else:
        print("no geometric correction on tt")
    if HWtth is not None and isinstance(HWtth, numbers.Number):
        correction_tth = np.radians(HWtth*2) / (np.arctan((np.tan(np.radians(gixs2d["tth"]))*Ddet + det_px/2)/Ddet) - np.arctan((np.tan(np.radians(gixs2d["tth"]))*Ddet - det_px/2)/Ddet))
    else:
        print("no geometric correction on tth")
    correction_matrix = np.outer(correction_tt, correction_tth)
    gixs2d["Intensity"] = gixs2d["Intensity"]*correction_matrix
    gixs2d["error"] = np.sqrt(gixs2d["Intensity"])
    return gixs2d

def GIXOS_th2q(inputdata):
    """
    create q axises from the angular axises
    Parameters
    ----------
    inputdata : dictionary
        required fields:
            'Intensity':    intensity map, 2d or one line cut,
            'tth':          tth axis in deg, 
            'tt':           tt axis in deg, 
            'metadata':     ['instrument'] with 'energy' (eV) and 'alpha' (deg)
    Returns
    -------
    outputdata : dictionary
        same field of inputdata
        additional fields:
            'Qxy':  (1/A)
            'Qz':   (1/A)
            'Q':    (1/A)

    """
    outputdata = None
    if (inputdata["metadata"] is None) or ("instrument" not in inputdata["metadata"]) or (inputdata["metadata"]["instrument"] is None) or (not check_keys_numeric(["energy", "alpha"], inputdata["metadata"]["instrument"])):
        print("please provide energy [eV] and incident angle (alpha) [deg] in the ['metadata']['instrument']")
        return
        
    inputdata['mat'] = inputdata.pop('Intensity')    
    # calculate qxy, qz, and q, use the th2q function from p08_GIXD, it requires the intensity to be called mat
    outputdata = th2q(inputdata, energy = inputdata["metadata"]["instrument"]["energy"], alpha_i = inputdata["metadata"]["instrument"]["alpha"], absQxy = False)
    outputdata["Q"] = np.sqrt(outputdata["Qxy"]**2 + outputdata["Qz"]**2)
    for key in inputdata.keys(): 
        if key not in ['mat', 'Qxy', 'Qz', 'Q']:
            outputdata[key] = inputdata[key]
    # swap back the key to intensity
    inputdata['Intensity'] = inputdata.pop('mat')
    outputdata['Intensity'] = outputdata.pop('mat')
    return outputdata

def extract_1dGIXOS(gixs2d, tth_array, HWpx_h = 5):
    """
    extract 1d GIXOS cut from the loaded 2d gixs image (in tth-tt corridnate), at given tth positions; the image pixel HW should be given

    Parameters
    ----------
    gixs2d : dictionary
        at least with Intensity, tt, tth, HWtth, HWtt, HWpx_h, HWpx_v, metadata
    tth_array : numpy array (1,n)
        positions of tth to extract GIXOS.
    HWpx_h : float, optional
        horizontal HW of the GIXOS linecut in pixel. Must be multiple of gixs2d['HWpx_h']. The default is 5.

    Returns
    -------
    gixos1d : dictionary
        GIXOS line cuts at the selected tth. Same as loaded 1d gixos data except for hosting multiple tth positions
        fields:
            'Intensity':    GIXOS linecuts,
            'error':        error
            'tth':          tth for each column in deg, 
            'tt':           tt axis in deg, 
            'HWtth':        half width for GIXOS cuts in tth in deg, 
            'HWtt':         half width for GIXOS data in tt in deg, 
            'HWpx_h':       horizontal half width in pixel, 
            'HWpx_v':       vertical half width in pixel.
            'metadata':     metadata          

    """
    gixos1d = None
    if tth_array.ndim == 1:  # means shape is (n,), tth should be changed into (1,n)
        tth_array = tth_array.reshape(1, -1)

    if abs(HWpx_h % gixs2d["HWpx_h"]) > 1e-9:
        print("the half width has to be multiple of the half width in the input data; check HWpx_h")
        return
    # build up output data structure, tt to be defined depending on the tth shape
    gixos1d = {
                "Intensity":    np.zeros((gixs2d["tt"].shape[0], tth_array.shape[1])),
                "error":        np.zeros((gixs2d["tt"].shape[0], tth_array.shape[1])),
                "tt":           None,
                "tth":          np.zeros((gixs2d["tth"].shape[0], tth_array.shape[1])),
                "HWtth":        HWpx_h/gixs2d["HWpx_h"]*gixs2d["HWtth"],
                "HWpx_h":       HWpx_h,
                "HWtt":         gixs2d["HWtt"],
                "HWpx_v":       gixs2d["HWpx_v"],
                "metadata":     gixs2d["metadata"]
                }            
    if gixs2d["tth"].shape[0] >1:
        # not exactly the horizon but rather okay when checking on the 0th column
        # when tth is two dimensional tt is also
        print("tt and tth are not rebined")
        gixos1d["tt"] = np.zeros((gixs2d["tt"].shape[0], tth_array.shape[1]))
        row_idx_horizon = np.abs(gixs2d["tt"][:,0]).argmin()
        col_idx_list = np.array([np.abs(gixs2d["tth"][row_idx_horizon,:] - tth).argmin() for tth in tth_array[0,:]])
    else:
        # when tt/tth are rebinned, such that they are both column/row vector array
        print("rebinned tt and tth, they are 1D arrays")
        gixos1d["tt"] = np.zeros((gixs2d["tt"].shape[0], 1))
        col_idx_list = np.array([np.abs(gixs2d["tth"] - tth).argmin() for tth in tth_array[0,:]])           
    
    for idx, col_idx in enumerate(col_idx_list):
        """
        populate the values
        """
        gixos1d["Intensity"][:, idx] = np.sum(gixs2d["Intensity"][:,col_idx-HWpx_h+1:col_idx+HWpx_h+1], axis = 1)
        if gixs2d["tth"].shape[0] >1:
            gixos1d["tt"][:, idx] = np.mean(gixs2d["tt"][:,col_idx-HWpx_h+1:col_idx+HWpx_h+1], axis = 1)
            gixos1d["tth"][:, idx] = np.mean(gixs2d["tth"][:,col_idx-HWpx_h+1:col_idx+HWpx_h+1], axis = 1)
        else:
            gixos1d["tt"] = gixs2d["tt"]
            gixos1d["tth"] = tth_array
    gixos1d["error"] = np.sqrt(gixos1d["Intensity"])
    
    return gixos1d
    

def binning_GIXOS_tt(GIXOSdata, HWpx_v = 5):
    """
    binning GIXOS data in vertical direction
    - originally work with both sample data and bkg
    - now make the general that only work with data

    Parameters
    ----------
    GIXOSdata : dictionary
        at least with Intensity, tt, tth, HWtth, HWtt, HWpx_h, HWpx_v, metadata
    HWpx_v : float, optional
        vertical HW of the binned GIXOS data in pixel. Must be multiple of GIXOS['HWpx_v']. The default is 5.

    Returns
    -------
    GIXOSdata : dictionary
        intensity, tt, HWtt, HWpx_v are updated, error is calculated.

    """
    binsize = HWpx_v*2
    groupnumber =  math.floor(GIXOSdata["Intensity"].shape[0] / binsize)      # look at the first row with .shape[0]
    num_columns = GIXOSdata["Intensity"].shape[1]
    
    # set the new matrix
    binneddata = {
        "Intensity":    np.zeros((groupnumber, num_columns)),
        "error":        np.zeros((groupnumber, num_columns)),
        "tt":           None,
        "tth":          None,
        "HWtth":        GIXOSdata["HWtth"],
        "HWpx_h":       GIXOSdata["HWpx_h"],
        "HWtt":         HWpx_v/GIXOSdata["HWpx_v"]*GIXOSdata["HWtt"],
        "HWpx_v":       HWpx_v,
        "metadata":     GIXOSdata["metadata"]
        } 
    # define tt according to the original matrix
    if GIXOSdata["tt"].ndim >1:
        binneddata["tt"] = np.zeros((groupnumber, num_columns))
    else:
        binneddata["tt"] = np.zeros((groupnumber))
    
    # populate the dictionary with values
    for groupidx in range(groupnumber): # why can't we just round up before if we are adding 1 to it?
        start = groupidx * binsize
        end = (groupidx + 1) * binsize
        binneddata ["Intensity"][groupidx, :] = np.sum(GIXOSdata ["Intensity"][start:end, :], axis=0)
        binneddata ["tt"][groupidx] = np.mean(GIXOSdata ["tt"][start:end])
        if "tth" in GIXOSdata:
            if GIXOSdata["tt"].ndim > 1:
                binneddata["tth"][groupidx, :] = np.mean(GIXOSdata ["tth"][start:end, :], axis=0)
            else:
                binneddata["tth"] = GIXOSdata["tth"]
        else:
            print("please include tth in the data['tth'] field!")
    binneddata["error"] = np.sqrt(binneddata["Intensity"])
    GIXOSdata = binneddata
    return GIXOSdata

def remove_negative_2theta(GIXOSdata):
    """
    remove the negative 2theta range

    Parameters
    ----------
    GIXOSdata : dictionary
        at least with Intensity, error, tt, tth.
        tt must start from negative and go ascending in idx
    Returns
    -------
    GIXOSdata : dictionary
        rows with negative tt are deleted.

    """
    indices =  np.where(GIXOSdata ["tt"] < 0)[0] # finding indices where value stored is less than 0
    tt_end_idx = indices[-1] if len(indices) > 0 else None  # taking the last  value of indices, and checking if indices is a valid list to take from
    if tt_end_idx is not None:
        GIXOSdata["Intensity"] = np.delete(GIXOSdata["Intensity"], np.s_[0:tt_end_idx+1], axis=0)
        GIXOSdata["error"] = np.delete(GIXOSdata["error"], np.s_[0:tt_end_idx+1], axis=0)
        if GIXOSdata["tt"].ndim > 1:
            GIXOSdata["tt"] = np.delete(GIXOSdata["tt"], np.s_[0:tt_end_idx+1], axis=0)
            GIXOSdata["tth"] = np.delete(GIXOSdata["tth"], np.s_[0:tt_end_idx+1], axis=0)
        else:
            GIXOSdata["tt"] = np.delete(GIXOSdata["tt"], np.s_[0:tt_end_idx+1], axis=0)
    return GIXOSdata


def GIXOS_background_corr(
                            sampledata,
                            chamberbkg,
                            bulkbkg_mode=None,
                            bulkbkg_const_mode=0,
                            bulkbkg_value=None,
                            bulkbkg_const_qz_lb=None,
                            bulkbkg_offset_lb=0.9,
                            bulkbkg_fit_qz_lb=None,
                        ):
    
    """
    Correct GIXOS data for chamber and bulk background.

    The correction is applied in two steps:

    1. Chamber background subtraction
       The chamber background is subtracted from the sample intensity.
       If flux and counting-time metadata are available, the chamber
       background is scaled accordingly before subtraction.

    2. Bulk (wide-angle) background subtraction
       A second background contribution can optionally be removed using
       one of several bulk-background models controlled by `bulkbkg_mode`.

    Parameters
    ----------
    sampledata : dict
        Sample GIXOS data dictionary. Required fields:
        - "Intensity"
        - "error"
        - "tt"
        - "tth"
        - "metadata"

        Some bulk-background modes also require:
        - "Q"
        - "Qz"
        - "Qxy"

    chamberbkg : dict or None
        Chamber background data dictionary with the same shape as `sampledata`.
        If None, chamber background is assumed to be zero.

    bulkbkg_mode : None or int, optional
        Bulk-background subtraction mode:

        - None : no bulk background subtraction
        - 0    : constant bulk background subtraction
        - 1    : direct subtraction using wide-angle column(s)
        - 2    : fit wide-angle background in Q and subtract the fitted curve

    bulkbkg_const_mode : int, optional
        Method used to determine the constant background when
        `bulkbkg_mode == 0`:

        - 0 : use user-provided `bulkbkg_value`
              This may be either:
              * a single scalar applied to all columns, or
              * an array-like of length m, giving one constant per column
        - 1 : determine one constant per column from the average intensity
              for data with Qz >= `bulkbkg_const_qz_lb`
        - 2 : determine one constant per column from the average of the
              three minimum intensities for data with Qz > 3*Qc

        Ignored unless `bulkbkg_mode == 0`.
        Default is 0.

    bulkbkg_value : float, optional
        Constant background value(s) to subtract when
        `bulkbkg_mode == 0` and `bulkbkg_const_mode == 0`.
        
        This can be either:
        - a scalar, applied to all columns
        - an array-like of length m, giving one constant per column

    bulkbkg_offset_lb : float, optional
        Lower-bound factor for the offset parameter y0 in the
        bulk-background fit. The lower bound is defined as

            mean(first 10 fit points) * bulkbkg_offset_lb

        Used only when `bulkbkg_mode == 2`.
        Default is 0.9.

    bulkbkg_fit_qz_lb : float or None, optional
        Lower Qz boundary for the fit region when `bulkbkg_mode == 2`.
        If None, the default internal lower-Q cutoff is used.

    bulkbkg_const_qz_lb : float or None, optional
        Lower Qz boundary used when
        `bulkbkg_mode == 0` and `bulkbkg_const_mode == 1`.
        The constant background is estimated from the average intensity
        above this Qz value.

    Returns
    -------
    correcteddata : dict
        Corrected GIXOS data dictionary. It preserves the non-intensity
        fields from `sampledata` and adds corrected:
        - "Intensity"
        - "error"

        If a bulk background is subtracted, a "bulkbkg" field is also added,
        containing the estimated or fitted bulk-background information.

    Notes
    -----
    Chamber-background normalization uses metadata if available:
    - sampledata["metadata"]["measurements"]["flux"]
    - sampledata["metadata"]["measurements"]["cttime_sample"]
    - chamberbkg["metadata"]["measurements"]["flux"]
    - chamberbkg["metadata"]["measurements"]["cttime_bkg"]

    For bulk-background subtraction:
    - modes 1 and 2 require:
      sampledata["metadata"]["qxy0"] and sampledata["metadata"]["qxy_bkg"]
    - mode 2 requires:
      sampledata["Q"]
    - constant mode 1 requires:
      sampledata["Qz"]
    - constant mode 2 requires:
      sampledata["Qz"] and sampledata["metadata"]["sample_params"]["Qc"]

    """    
    # ------------------------------------------------------------
    # validate bulk-background mode and related inputs
    # ------------------------------------------------------------
    if bulkbkg_mode not in (None, 0, 1, 2):
        raise ValueError("bulkbkg_mode must be one of None, 0, 1, 2.")

    if bulkbkg_mode == 0:
        if bulkbkg_const_mode not in (0, 1, 2):
            raise ValueError(
                "bulkbkg_const_mode must be one of 0, 1, 2 when bulkbkg_mode == 0."
            )

        if bulkbkg_const_mode == 0:
            if bulkbkg_value is None:
                raise ValueError(
                    "bulkbkg_value must be provided when bulkbkg_mode == 0 "
                    "and bulkbkg_const_mode == 0."
                )

            ncols = sampledata["Intensity"].shape[1]

            if np.ndim(bulkbkg_value) == 0:
                try:
                    bulkbkg_value = float(bulkbkg_value)
                except (TypeError, ValueError):
                    raise ValueError("bulkbkg_value must be numeric.")
            else:
                try:
                    bulkbkg_value = np.asarray(bulkbkg_value, dtype=float).ravel()
                except (TypeError, ValueError):
                    raise ValueError("bulkbkg_value must be numeric.")

                if bulkbkg_value.shape[0] != ncols:
                    raise ValueError(
                        f"When bulkbkg_value is array-like, it must have length {ncols} "
                        "(one value per column)."
                    )

        elif bulkbkg_const_mode == 1:
            if "Qz" not in sampledata:
                raise ValueError(
                    "sampledata['Qz'] is required when bulkbkg_mode == 0 "
                    "and bulkbkg_const_mode == 1."
                )
            if bulkbkg_const_qz_lb is None:
                raise ValueError(
                    "bulkbkg_const_qz_lb must be provided when bulkbkg_const_mode == 1."
                )
            if np.ndim(bulkbkg_const_qz_lb) != 0:
                raise ValueError("bulkbkg_const_qz_lb must be a scalar.")
            try:
                bulkbkg_const_qz_lb = float(bulkbkg_const_qz_lb)
            except (TypeError, ValueError):
                raise ValueError("bulkbkg_const_qz_lb must be numeric.")

        elif bulkbkg_const_mode == 2:
            if "Qz" not in sampledata:
                raise ValueError(
                    "sampledata['Qz'] is required when bulkbkg_mode == 0 "
                    "and bulkbkg_const_mode == 2."
                )
            if (
                "metadata" not in sampledata or
                sampledata["metadata"] is None or
                "sample_params" not in sampledata["metadata"] or
                "Qc" not in sampledata["metadata"]["sample_params"] or
                sampledata["metadata"]["sample_params"]["Qc"] is None
            ):
                raise ValueError(
                    "sampledata['metadata']['sample_params']['Qc'] is required "
                    "when bulkbkg_const_mode == 2."
                )

    elif bulkbkg_mode == 1:
        if (
            "metadata" not in sampledata or
            sampledata["metadata"] is None or
            "qxy0" not in sampledata["metadata"] or
            "qxy_bkg" not in sampledata["metadata"]
        ):
            raise ValueError(
                "sampledata['metadata']['qxy0'] and sampledata['metadata']['qxy_bkg'] "
                "are required when bulkbkg_mode == 1."
            )

    elif bulkbkg_mode == 2:
        if (
            "metadata" not in sampledata or
            sampledata["metadata"] is None or
            "qxy0" not in sampledata["metadata"] or
            "qxy_bkg" not in sampledata["metadata"]
        ):
            raise ValueError(
                "sampledata['metadata']['qxy0'] and sampledata['metadata']['qxy_bkg'] "
                "are required when bulkbkg_mode == 2."
            )
        if "Q" not in sampledata:
            raise ValueError("sampledata['Q'] is required when bulkbkg_mode == 2.")

        if bulkbkg_fit_qz_lb is not None:
            if np.ndim(bulkbkg_fit_qz_lb) != 0:
                raise ValueError("bulkbkg_fit_qz_lb must be a scalar.")
            try:
                bulkbkg_fit_qz_lb = float(bulkbkg_fit_qz_lb)
            except (TypeError, ValueError):
                raise ValueError("bulkbkg_fit_qz_lb must be numeric.")
                
    # -------------------------------------------------------------------------
    # start background correction
    # -------------------------------------------------------------------------
    # set disctionary 
    correcteddata = {   
                        "Intensity": None
                    }
    bulkbkg = {   
                        "Intensity": None
                    }
    
    # -------------------------------------------------------------------------
    # chamber background ubstraction
    # -------------------------------------------------------------------------
    I0_sample_bkg = 1.0
    if chamberbkg is None:
        chamberbkg["Intensity"] = np.zeros((sampledata["Intensity"].shape[0], sampledata["Intensity"].shape[1]))
    
    # only the same shape can be treated
    if sampledata["Intensity"].shape != chamberbkg["Intensity"].shape:
        print("sampledata intensity matrix must have the same shape as the chamber bkg intensity matrix")
        return
        
    # except for the intensity and error to be calculated, all others should be passed to the result 
    for key in sampledata.keys() - ["Intensity", "error"]:
        correcteddata[key] = sampledata[key]
    
    # normalisation factor for the integrated flux
    if check_keys_numeric(["flux", "cttime_sample"], sampledata["metadata"]["measurements"]) and check_keys_numeric(["flux", "cttime_bkg"], chamberbkg["metadata"]["measurements"]):
        I0_sample_bkg = sampledata["metadata"]["measurements"]["flux"]*sampledata["metadata"]["measurements"]["cttime_sample"] / (chamberbkg["metadata"]["measurements"]["flux"]*chamberbkg["metadata"]["measurements"]["cttime_bkg"])
    else:
        print("sample and chamber bkg are considered to have the same flux and counting time")
    
    # subtract chamber bkg
    data_chamber_subtracted = sampledata["Intensity"] - chamberbkg["Intensity"]*I0_sample_bkg
    err_propogate = np.sqrt(sampledata["error"]**2 + chamberbkg["error"]**2 * I0_sample_bkg**2)
                    
    # -------------------------------------------------------------------------
    # bulk background subtraction
    # -------------------------------------------------------------------------
    if bulkbkg_mode is None:
        # no bulk subtraction
        print("no bulkbkg subtraction")
        correcteddata["Intensity"] = data_chamber_subtracted
        correcteddata["error"] = err_propogate
        
    elif bulkbkg_mode == 0:
        # constant background
        if bulkbkg_const_mode == 0:
            # user-provided scalar
            print("constant bulk background: user-provided value")

            nrows, ncols = data_chamber_subtracted.shape

            if np.ndim(bulkbkg_value) == 0:
                bulk_const = np.full(ncols, float(bulkbkg_value), dtype=float)
            else:
                bulk_const = np.asarray(bulkbkg_value, dtype=float).ravel()

            bulkbkg_2d = np.outer(np.ones(nrows), bulk_const)   # shape (n, m)

            correcteddata["Intensity"] = data_chamber_subtracted - bulkbkg_2d
            correcteddata["error"] = err_propogate

            bulkbkg["Intensity"] = bulk_const[None, :]          # shape (1, m)
            bulkbkg["Intensity_at_GIXOS"] = bulkbkg_2d
            bulkbkg["const_mode"] = 0
            correcteddata["bulkbkg"] = bulkbkg
            
        elif bulkbkg_const_mode == 1:
            # mean above Qz threshold
            
            print("constant bulk background: average above Qz lower bound")
            
            qz_data = sampledata["Qz"]
            if qz_data.ndim == 1:
                qz_data = qz_data[:, None]
            
            ncols = data_chamber_subtracted.shape[1]
            bulk_const = np.zeros(ncols)
            
            for j in range(ncols):
                mask = qz_data[:, j] >= bulkbkg_const_qz_lb
                if np.count_nonzero(mask) == 0:
                    raise ValueError(
                        f"No data points satisfy Qz >= {bulkbkg_const_qz_lb} for column {j}."
                    )
                bulk_const[j] = np.mean(data_chamber_subtracted[mask, j])
            
            bulkbkg_2d = np.outer(np.ones(data_chamber_subtracted.shape[0]), bulk_const)

            bulkbkg["Intensity"] = bulk_const[None, :]   # shape (1, m)
            bulkbkg["Intensity_at_GIXOS"] = bulkbkg_2d
            bulkbkg["const_mode"] = 1
            bulkbkg["const_qz_lb"] = bulkbkg_const_qz_lb
            
            correcteddata["Intensity"] = data_chamber_subtracted - bulkbkg["Intensity_at_GIXOS"]
            correcteddata["error"] = err_propogate
            correcteddata["bulkbkg"] = bulkbkg
            
        elif bulkbkg_const_mode == 2:
            # mean of 3 minima above 3*Qc
            
            print("constant bulk background: mean of 3 minima above 3Qc")
            
            qz_data = sampledata["Qz"]
            if qz_data.ndim == 1:
                qz_data = qz_data[:, None]
            
            qz_lb = 3.0 * sampledata["metadata"]["sample_params"]["Qc"]
            ncols = data_chamber_subtracted.shape[1]
            bulk_const = np.zeros(ncols)
            
            for j in range(ncols):
                mask = qz_data[:, j] > qz_lb
                vals = data_chamber_subtracted[mask, j]
                if vals.size < 3:
                    raise ValueError(
                        f"Fewer than 3 points satisfy Qz > 3Qc for column {j}."
                    )
                bulk_const[j] = np.mean(np.sort(vals)[:3])
            
            bulkbkg_2d = np.outer(np.ones(data_chamber_subtracted.shape[0]), bulk_const)

            bulkbkg["Intensity"] = bulk_const[None, :]   # shape (1, m)
            bulkbkg["Intensity_at_GIXOS"] = bulkbkg_2d
            bulkbkg["const_mode"] = 2
            bulkbkg["const_qz_lb"] = qz_lb
            
            correcteddata["Intensity"] = data_chamber_subtracted - bulkbkg["Intensity_at_GIXOS"]
            correcteddata["error"] = err_propogate
            correcteddata["bulkbkg"] = bulkbkg
    
    else:
        # if wide angle data exist, subtract the wide angle, either directly using line cut, or using fit over q
        qxy0_idx_arr = np.where(sampledata["metadata"]["qxy0"] > sampledata["metadata"]["qxy_bkg"])[0]  # get the array of the wide angle column
        if len(qxy0_idx_arr) == 0:
            print("bkg qxy0 is larger than the largest qxy0 position. No bulk bkg subtraction")
            correcteddata["Intensity"] = data_chamber_subtracted
            correcteddata["error"] = err_propogate
            return        
        else:
            bulk_qxy0_idx = qxy0_idx_arr
            bulkbkg["Intensity"] = np.mean(np.atleast_2d(data_chamber_subtracted[:, bulk_qxy0_idx]), axis = 1)   # average the wide angle intensity over those qxy0
            bulkbkg["error"] = np.sqrt( np.sum(np.atleast_2d(err_propogate[:, bulk_qxy0_idx]**2), axis = 1) ) /len(bulk_qxy0_idx)
            # populate the axises for the bulkbkg
            bulkbkg["tth"] = np.mean(np.atleast_2d(correcteddata["tth"][0,bulk_qxy0_idx]), axis = 1)
            # tt can be a vector array (rebinned) or a matrix (not rebinned)
            if correcteddata["tt"].ndim>1:
                bulkbkg["tt"] = np.mean(np.atleast_2d(correcteddata["tt"][:,bulk_qxy0_idx]), axis = 1)
            else:
                bulkbkg["tt"] = correcteddata["tt"]
            if "Q" in correcteddata:
                # because it is not necessarily required to have Q axises
                bulkbkg["Qxy"] = np.mean(np.atleast_2d(correcteddata["Qxy"][:,bulk_qxy0_idx]), axis = 1)
                bulkbkg["Qz"] = np.mean(np.atleast_2d(correcteddata["Qz"][:,bulk_qxy0_idx]), axis = 1)
                bulkbkg["Q"] = np.mean(np.atleast_2d(correcteddata["Q"][:,bulk_qxy0_idx]), axis = 1)
            
            # two modes of bkg
            if bulkbkg_mode == 1:
                # direct wide-angle subtraction
                bulkbkg["Intensity_at_GIXOS"] = np.outer(bulkbkg["Intensity"],np.ones((1,bulk_qxy0_idx[0])))
                correcteddata["Intensity"] = data_chamber_subtracted[:,:bulk_qxy0_idx[0]] - bulkbkg["Intensity_at_GIXOS"]
                correcteddata["error"] = np.sqrt(err_propogate[:,:bulk_qxy0_idx[0]]**2 + (np.outer(bulkbkg["error"],np.ones((1,bulk_qxy0_idx[0]))))**2)
                correcteddata["bulkbkg"] = bulkbkg
            else:
                # mode 2: fit wide-angle in Q
                if "Q" in correcteddata:
                    bulkbkg_Q = np.mean(np.atleast_2d(correcteddata["Q"][:, bulk_qxy0_idx]), axis = 1) # this is the q axis
                    bulkbkg_Qz = np.mean(np.atleast_2d(correcteddata["Qz"][:, bulk_qxy0_idx]), axis=1)
                    # choose fit range:
                    # - default: exclude the lowest 10% of the Q range
                    # - optional: start fitting from user-defined lower Qz boundary that is larger than the lowest 10%
                    Q_cut = np.min(bulkbkg_Q) + 0.1 * (np.max(bulkbkg_Q) - np.min(bulkbkg_Q))
                    
                    if bulkbkg_fit_qz_lb is None:
                        mask = bulkbkg_Q >= Q_cut
                    else:
                        mask = (bulkbkg_Q >= Q_cut) & (bulkbkg_Qz >= bulkbkg_fit_qz_lb)
                    
                    if np.count_nonzero(mask) < 3:
                        raise ValueError("Not enough points remain in the selected bulk background fit range.")
                    
                    y0_lb = np.mean(bulkbkg["Intensity"][mask][:10], axis=0) * bulkbkg_offset_lb
                    # fit Q
                    res = bulkbkg_fit(
                                        bulkbkg_Q[mask], 
                                        bulkbkg["Intensity"][mask], 
                                        y0_bounds=(y0_lb, np.inf)
                                        )
                    bulkbkg_y0, bulkbkg_F, bulkbkg_t = res["popt"]
                    print("y0: %f\nF: %f\nt: %f\n" %(bulkbkg_y0, bulkbkg_F, bulkbkg_t))
                    # optional plot
                    bulkbkg_plot_fit(res)
                    # load result into the bulkbkg
                    bulkbkg["fit_params"] = {'y0': bulkbkg_y0, 'F': bulkbkg_F, 't': bulkbkg_t}
                    bulkbkg['Intensity_at_GIXOS'] = bulkbkg_predict(correcteddata["Q"][:,:bulk_qxy0_idx[0]], res['popt'])
                    # subtract bulk bkg for every GIXOS cut
                    correcteddata["Intensity"] = data_chamber_subtracted[:,:bulk_qxy0_idx[0]] - bulkbkg['Intensity_at_GIXOS']
                    correcteddata["error"] = err_propogate[:,:bulk_qxy0_idx[0]]
                    
                    correcteddata['bulkbkg'] = bulkbkg
                else:
                    print("input data requires Q axis")

            correcteddata["tth"] = np.delete(correcteddata["tth"], np.s_[bulk_qxy0_idx], axis=1)
            if correcteddata["tt"].ndim>1:
                correcteddata["tt"] = np.delete(correcteddata["tt"], np.s_[bulk_qxy0_idx], axis=1)
            if "Q" in correcteddata:
                correcteddata["Qxy"] = np.delete(correcteddata["Qxy"], np.s_[bulk_qxy0_idx], axis=1)
                correcteddata["Qz"] = np.delete(correcteddata["Qz"], np.s_[bulk_qxy0_idx], axis=1)
                correcteddata["Q"] = np.delete(correcteddata["Q"], np.s_[bulk_qxy0_idx], axis=1)
        
        
    return correcteddata
       

#%% analysis with eCWM
def GIXOS_qxy_dependence(
    GIXOSdict,
    qz_targets,
    *,
    fit_kappa=False,
    row_window=1,          # +/- rows around center (1 -> 3 rows total)
    offset_factor=100.0,   # curve k is multiplied by offset_factor**k
    normalize_point_index=2,
    plot=True,
):
    
    """
    Analyze the Qxy dependence of GIXOS intensity at selected Qz positions.

    This function extracts horizontal line cuts of diffuse scattering intensity
    as a function of Qxy at one or more selected Qz values, and compares them
    with diffuse-scattering roughness-factor predictions from the capillary wave
    model (CWM) and the extended capillary wave model (eCWM).

    For each target Qz value, the nearest row in the GIXOS map is selected.
    The intensity can optionally be summed over neighboring rows
    (`row_window`) to improve statistics.

    Two modes are supported:

    - Prediction mode (`fit_kappa=False`)
        Use the bending rigidity `kappa` stored in
        `GIXOSdict['metadata']['sample_params']['kappa']` and calculate the
        corresponding eCWM prediction.

    - Fit mode (`fit_kappa=True`)
        Fit a single global bending rigidity `kappa` to all selected Qz cuts
        simultaneously in log10-log10 space.

    In both cases, the calculated model curves are normalized row-by-row to the
    measured data at `normalize_point_index` so that the comparison focuses on
    the Qxy dependence (shape) rather than the absolute scale.

    Parameters
    ----------
    GIXOSdict : dict
        Processed GIXOS dataset dictionary. Required fields:
        - "Intensity" : 2D intensity array, shape (nrows, ncols)
        - "Qxy"       : 2D Qxy array, same shape as Intensity
        - "Qz"        : 2D Qz array, same shape as Intensity
        - "tt"        : beta / exit-angle axis
        - "tth"       : phi / in-plane scattering angle axis
        - "HWtt"      : half width of one tt bin
        - "HWtth"     : half width of one tth bin
        - "metadata"

        Required metadata fields:
        - GIXOSdict["metadata"]["instrument"]["energy"]
        - GIXOSdict["metadata"]["instrument"]["alpha"]
        - GIXOSdict["metadata"]["sample_params"]["temperature"]
        - GIXOSdict["metadata"]["sample_params"]["tension"]
        - GIXOSdict["metadata"]["sample_params"]["amin"]
        - GIXOSdict["metadata"]["sample_params"]["kappa"]

    qz_targets : array-like
        One or more target Qz values (Å⁻¹) at which the Qxy dependence should
        be analyzed. Each target is matched to the nearest available row in the
        GIXOS map.

    fit_kappa : bool, optional
        If False (default), calculate the eCWM prediction using the `kappa`
        value already stored in the metadata.

        If True, fit one global `kappa` value to all selected Qz cuts
        simultaneously in log10-log10 space.

    row_window : int, optional
        Number of neighboring rows on each side of the selected row to include
        in the summed intensity. The total number of rows summed is
        `2 * row_window + 1`.

        For example:
        - row_window = 0 → only the nearest row
        - row_window = 1 → nearest row ±1 (3 rows total)

        Default is 1.

    offset_factor : float, optional
        Multiplicative offset applied between different extracted Qz cuts for
        visualization in the plot. This affects only the plotted offset curves,
        not the fitting or normalization.

        Default is 100.0.

    normalize_point_index : int, optional
        Column index in the extracted Qxy dependence used to normalize the
        CWM/eCWM model curves to the measured data for each selected Qz cut.

        This is applied row-by-row and is used both in prediction mode and in
        fit mode.

        Default is 2.

    plot : bool, optional
        If True, plot the extracted Qxy dependences together with the CWM and
        eCWM reference curves.

        Default is True.

    Returns
    -------
    GIXOSdict : dict
        Input dictionary with additional analysis results written back into:
        - GIXOSdict["qxy_dependence_ana"]
        - GIXOSdict["metadata"]["sample_params"]["kappa"]
        - GIXOSdict["metadata"]["sample_params"]["kappa_err"]
        - GIXOSdict["metadata"]["sample_params"]["kappa_fit_success"]

    results : dict
        Dictionary containing the extracted data and model comparisons.

        Main fields include:
        - "target_qz"       : requested Qz target values
        - "row_index"       : selected row indices
        - "Qxy"             : extracted Qxy values for each target row
        - "I_sum"           : summed measured intensity for each target row
        - "I_sum_offset"    : offset version of I_sum for plotting
        - "DS_CWM"          : diffuse-scattering roughness factor for CWM
        - "DS_eCWM"         : diffuse-scattering roughness factor for eCWM
        - "ref_CWM"         : normalized CWM reference curves
        - "ref_eCWM"        : normalized eCWM reference curves
        - "fit_kappa"       : fitted or used kappa value
        - "fit_kappa_err"   : estimated 1σ uncertainty of fitted kappa
        - "fit_success"     : whether the fit converged
        - "fit_message"     : optimizer message
        - "fit_cost"        : least-squares cost function value

        If fitting is enabled, the following are also included:
        - "DS_eCWM_err_u"
        - "DS_eCWM_err_l"
        - "ref_eCWM_err_u"
        - "ref_eCWM_err_l"

    Notes
    -----
    The diffuse-scattering reference curves are calculated using
    `calc_eCWM_roughness_factor_DS(...)` at the selected beta (tt) and phi
    (tth) values.

    In fit mode, the fitting is performed on:

        log10(I_model) - log10(I_data)

    over all valid data points from all selected Qz cuts simultaneously.

    The fitted parameter is a single shared `kappa` value, while the row-wise
    normalization factor is determined independently for each selected Qz cut
    using `normalize_point_index`.

    This analysis is intended to compare the *shape* of the Qxy dependence
    rather than the absolute intensity scale.
    """

    # ---- parameters ----
    temperature = GIXOSdict['metadata']['sample_params']['temperature']
    tension = GIXOSdict['metadata']['sample_params']['tension']
    amin = GIXOSdict['metadata']['sample_params']['amin']
    kappa = GIXOSdict['metadata']['sample_params']['kappa']

    energy = GIXOSdict['metadata']['instrument']['energy']
    alpha = GIXOSdict['metadata']['instrument']['alpha']
    HWtth = GIXOSdict['HWtth'][0, 0]
    HWtt = GIXOSdict['HWtt'][0]

    # ---- reshape per-cell coordinates to grids matching intensity ----
    I = GIXOSdict["Intensity"]
    Qxy = GIXOSdict["Qxy"].reshape(I.shape)
    Qz = GIXOSdict["Qz"].reshape(I.shape)
    beta_space = GIXOSdict["tt"]
    phi = GIXOSdict["tth"][0, :]

    nrows, ncols = I.shape

    # representative Qz per row
    row_qz = np.median(Qz, axis=1)

    # allow qz_targets to be list or np array
    qz_targets = np.asarray(qz_targets, dtype=float).ravel()

    results = {
        'target_qz': qz_targets,
        'row_index': [],
        'Qxy': np.empty((0, len(phi))),
        'I_sum': np.empty((0, len(phi))),
        'I_sum_offset': np.empty((0, len(phi))),
        'DS_CWM': None,
        'DS_eCWM': None,
        'ref_CWM': None,
        'ref_eCWM': None,
        'fit_kappa': None,
        'fit_success': None,
        'fit_message': None,
        'fit_cost': None,
    }

    # ---- extract data for requested qz values ----
    for k_idx, qz_t in enumerate(qz_targets):
        i0 = int(np.argmin(np.abs(row_qz - qz_t)))

        i_lo = max(0, i0 - row_window)
        i_hi = min(nrows - 1, i0 + row_window)
        rows = np.arange(i_lo, i_hi + 1)

        qxy = np.mean(Qxy[rows, :], axis=0)
        y_sum = np.sum(I[rows, :], axis=0)
        y_off = y_sum * (offset_factor ** k_idx)

        results['row_index'].append(i0)
        results['Qxy'] = np.vstack([results['Qxy'], qxy])
        results['I_sum'] = np.vstack([results['I_sum'], y_sum])
        results['I_sum_offset'] = np.vstack([results['I_sum_offset'], y_off])

    row_index = np.asarray(results['row_index'], dtype=int)
    y_data = np.asarray(results['I_sum'], dtype=float)
    yoff_data = np.asarray(results['I_sum_offset'], dtype=float)
    nsets, nphi = y_data.shape
    norm_idx = int(np.clip(normalize_point_index, 0, nphi - 1))

    # ---- shared model builder ----
    def build_ds_for_kappa(kappa_value):
        ds_cols = []
        
        for phi_value in phi:            
            ds_model = calc_eCWM_roughness_factor_DS(
                alpha,
                beta_space[row_index],
                phi_value,
                energy = energy,
                DSphi_HWHM = HWtth,
                DSbeta_HWHM = HWtt * (row_window * 2 + 1),
                tension = tension,
                temp = temperature,
                kappa = kappa_value,
                amin = amin,
                use_approx=True
                )
            
            ds_cols.append(np.asarray(ds_model, dtype=float))
        return np.column_stack(ds_cols)   # (nsets, nphi)

    # ---- CWM reference is always built ----
    results['DS_CWM'] = build_ds_for_kappa(0.0)

    # ---- choose eCWM kappa: fitted or metadata ----
    if fit_kappa:
        from scipy.optimize import least_squares

        def residuals_log10(p):
            kappa_trial = float(p[0])

            if kappa_trial < 0:
                return np.full(y_data.size, 1e6, dtype=float)

            y_model_base = build_ds_for_kappa(kappa_trial)

            valid = (
                np.isfinite(y_data) & (y_data > 0) &
                np.isfinite(y_model_base) & (y_model_base > 0)
            )

            if not np.all(valid[:, norm_idx]):
                return np.full(y_data.size, 1e6, dtype=float)

            F_row = y_data[:, norm_idx] / y_model_base[:, norm_idx]
            y_model = y_model_base * F_row[:, None]

            good = (
                np.isfinite(y_model) & (y_model > 0) &
                np.isfinite(y_data) & (y_data > 0)
            )

            if not np.any(good):
                return np.full(y_data.size, 1e6, dtype=float)

            return (np.log10(y_model[good]) - np.log10(y_data[good])).ravel()

        fitres = least_squares(
            residuals_log10,
            x0=np.array([kappa], dtype=float),
            bounds=(0.0, np.inf)
        )

        kappa_use = float(fitres.x[0])
        # --- error bar ----
        J = fitres.jac  # shape (ndata, nparams)
        res = fitres.fun  # residuals
        # number of data points and parameters
        n = len(res)
        p = J.shape[1]
        # residual variance (reduced chi^2 estimate)
        s_sq = np.sum(res**2) / (n - p)
        # covariance matrix
        cov = s_sq * np.linalg.pinv(J.T @ J)
        # standard error of kappa (only 1 parameter here)
        kappa_err = np.sqrt(cov[0, 0])
        
        results['fit_kappa'] = kappa_use
        results['fit_kappa_err'] = kappa_err
        results['fit_success'] = fitres.success
        results['fit_message'] = fitres.message
        results['fit_cost'] = fitres.cost

    else:
        kappa_use = float(kappa)
        results['fit_kappa'] = kappa_use
        results['fit_kappa_err'] = 0

    # ---- build eCWM model with chosen kappa ----
    results['DS_eCWM'] = build_ds_for_kappa(kappa_use)
    if fit_kappa:
        results['DS_eCWM_err_u'] = build_ds_for_kappa(kappa_use + kappa_err)
        results['DS_eCWM_err_l'] = build_ds_for_kappa(np.maximum(kappa_use - kappa_err,0))
    # ---- normalize both model references in one shared way ----
    y0 = yoff_data[:, norm_idx]

    A = y0 / results['DS_CWM'][:, norm_idx]
    B = y0 / results['DS_eCWM'][:, norm_idx]

    results['ref_CWM'] = results['DS_CWM'] * A[:, None]
    results['ref_eCWM'] = results['DS_eCWM'] * B[:, None]
    if fit_kappa:
        results['ref_eCWM_err_u'] = results['DS_eCWM_err_u'] * B[:, None]
        results['ref_eCWM_err_l'] = results['DS_eCWM_err_l'] * B[:, None]

    # ---- plot once ----
    if plot:
        if fit_kappa:
            fig_qxy, ax_qxy = GIXOS_qxy_dependence_plot(
                results,
                metadata=GIXOSdict["metadata"],
                title = rf"$Q_{{xy}}$ dependence fit, $\kappa = {kappa_use:.0f}\pm{kappa_err:.0f}\,k_{{\mathrm{{B}}}}T$"
            )
        else:
            fig_qxy, ax_qxy = GIXOS_qxy_dependence_plot(
                results,
                metadata=GIXOSdict["metadata"],
                title = rf"$Q_{{xy}}$ dependence predict, $\kappa = {kappa_use:.0f}\,k_{{B}}T$"
            )
        plotfilename = make_filename(GIXOSdict["metadata"], suffix = 'Qxy.png')
        fig_qxy.savefig(plotfilename, dpi=300, bbox_inches="tight")
    
    GIXOSdict['qxy_dependence_ana'] = results
    GIXOSdict["metadata"]["sample_params"]["kappa"] =  results['fit_kappa']
    GIXOSdict["metadata"]["sample_params"]["kappa_err"] =  results['fit_kappa_err']
    GIXOSdict["metadata"]["sample_params"]["kappa_fit_success"] =  results['fit_success']
    
    return GIXOSdict, results


# processing into SF and RRF
def GIXOS2R(GIXOS, transmission_corr = False, footprint_effect = False, use_approx = False, plot=True):
    """
    Convert GIXOS intensity into pseudo-reflectivity and structure factor.

    This function converts a processed GIXOS dataset into:

    - pseudo-reflectivity R(Qz)
    - intrinsic structure factor |Phi(Qz)|^2

    using the extended capillary wave model (eCWM). The conversion combines:

    1. Fresnel reflectivity of the substrate/interface
    2. optional transmission correction for the exit beam
    3. optional footprint-induced Qz broadening
    4. the eCWM roughness factors for diffuse and specular scattering

    Internally, the function evaluates the reduced roughness-factor ratio
    r_red = Psi_DS / Psi_R using `calc_eCWM_red_r(...)`, then uses it to
    transform the measured diffuse scattering intensity into a pseudo-XRR
    reflectivity and structure factor.

    Parameters
    ----------
    GIXOS : dict
        GIXOS dataset dictionary. It must contain at least:
        - "Intensity"
        - "error"
        - "Qz"
        - "tt"
        - "tth"
        - "HWtt"
        - "HWtth"
        - "metadata"

        The embedded metadata must contain the sections:
        - ["instrument"]
        - ["sample_params"]
        - ["PseudoR"]

    transmission_corr : bool, optional
        If True, apply transmission correction for the exit angle beta
        using `calc_tbeta_sqr()`. Default is False.

    footprint_effect : bool, optional
        If True, calculate footprint-induced Qz broadening using
        `GIXOS_dQz()`. Default is False.

    use_approx : bool, optional
        If True, use the approximate eCWM form in the roughness-factor
        calculation. If False, use the more complete / accurate form.
        Default is False.
    
    plot : bool, optional
        If True, plot the R/RF and R with SF, GIXOS and Psi_R.

        Default is True.
        
    Returns
    -------
    GIXOS : dict
        The input dictionary with additional fields added, including:
        - "fresnel"
        - "talpha_sqr"
        - "tbeta_sqr"
        - "dQz"
        - "r_reduced"
        - "Psi_DS"
        - "Psi_R"
        - "prefactor_DS"
        - "refl"
        - "SF"

    Notes
    -----
    The following metadata entries are required for the eCWM conversion:

    In GIXOS["metadata"]["instrument"]:
        - "alpha"   : incident angle in degrees
        - "energy"  : x-ray energy in eV
        - "Ddet"    : detector distance in mm (for some corrections)
        - "footprint" : footprint in mm (if footprint_effect=True)

    In GIXOS["metadata"]["sample_params"]:
        - "Qc"          : critical Q
        - "tension"     : surface tension [N/m]
        - "temperature" : temperature [K]
        - "kappa"       : bending rigidity [k_B T]
        - "amin"        : molecular cutoff length [Å]

    In GIXOS["metadata"]["PseudoR"]:
        - "qxy0_select_idx"
        - "resolution_mode"
        - "resolution_HW"
        - "energy"
        - "Ddet"
        - "bkg_mode"
        - "bkg_off"

    The pseudo-reflectivity and structure factor are derived from the
    measured diffuse scattering intensity and should be interpreted within
    the eCWM formalism used here.
    """
    
    # ------------------------------------------------------------
    # Validate required metadata for pseudo-XRR / eCWM conversion
    # ------------------------------------------------------------
    if "metadata" not in GIXOS or GIXOS["metadata"] is None:
        raise ValueError("GIXOS['metadata'] is required.")

    meta = GIXOS["metadata"]

    for section in ["instrument", "sample_params", "PseudoR"]:
        if section not in meta or meta[section] is None:
            raise ValueError(f"GIXOS['metadata']['{section}'] is required.")

    inst = meta["instrument"]
    samp = meta["sample_params"]
    pr = meta["PseudoR"]

    # optional keys that may not exist in metadata
    pr.setdefault("energy", None)
    pr.setdefault("Ddet", None)
    pr.setdefault("bkg_mode", None)
    pr.setdefault("bkg_off", None)

    # always required in instrument
    required_instrument = ["alpha", "energy"]

    # always required in sample_params
    required_sample = ["Qc", "tension", "temperature", "kappa", "amin"]

    # always required in PseudoR
    required_pseudor = ["qxy0_select_idx", "resolution_mode", "resolution_HW"]

    for key in required_instrument:
        if key not in inst or inst[key] is None:
            raise ValueError(f"GIXOS['metadata']['instrument']['{key}'] is required.")

    for key in required_sample:
        if key not in samp or samp[key] is None:
            raise ValueError(f"GIXOS['metadata']['sample_params']['{key}'] is required.")

    for key in required_pseudor:
        if key not in pr or pr[key] is None:
            raise ValueError(f"GIXOS['metadata']['PseudoR']['{key}'] is required.")

    # integer index check
    try:
        pr["qxy0_select_idx"] = int(pr["qxy0_select_idx"])
    except (TypeError, ValueError):
        raise ValueError("GIXOS['metadata']['PseudoR']['qxy0_select_idx'] must be an integer.")

    # scalar numeric checks for always-required values
    scalar_checks = {
        "instrument.alpha": inst["alpha"],
        "instrument.energy": inst["energy"],
        "sample_params.Qc": samp["Qc"],
        "sample_params.tension": samp["tension"],
        "sample_params.temperature": samp["temperature"],
        "sample_params.kappa": samp["kappa"],
        "sample_params.amin": samp["amin"],
    }

    for name, value in scalar_checks.items():
        if np.ndim(value) != 0:
            raise ValueError(f"{name} must be a scalar.")
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric.")

    # ----------------------------------------
    # PseudoR conditional requirements
    # ----------------------------------------
    if pr["resolution_mode"] not in (0, 1):
        raise ValueError("GIXOS['metadata']['PseudoR']['resolution_mode'] must be 0 or 1.")

    if pr["bkg_mode"] not in (None, 0, 1):
        raise ValueError("GIXOS['metadata']['PseudoR']['bkg_mode'] must be None, 0, or 1.")

    # resolution depends on resolution_mode
    if pr["resolution_mode"] == 0:
        if np.ndim(pr["resolution_HW"]) != 0:
            raise ValueError(
                "For PseudoR resolution_mode == 0, metadata['PseudoR']['resolution_HW'] must be a scalar."
            )
        try:
            pr["resolution_HW"] = float(pr["resolution_HW"])
        except (TypeError, ValueError):
            raise ValueError("metadata['PseudoR']['resolution_HW'] must be numeric.")

    elif pr["resolution_mode"] == 1:
        if pr["energy"] is None:
            raise ValueError(
                "metadata['PseudoR']['energy'] is required when resolution_mode == 1."
            )
        if pr["Ddet"] is None:
            raise ValueError(
                "metadata['PseudoR']['Ddet'] is required when resolution_mode == 1."
            )

        res = np.asarray(pr["resolution_HW"], dtype=float)
        if res.shape != (2,):
            raise ValueError(
                "For PseudoR resolution_mode == 1, metadata['PseudoR']['resolution_HW'] must be length 2."
            )
        pr["resolution_HW"] = res

        for key in ["energy", "Ddet"]:
            if np.ndim(pr[key]) != 0:
                raise ValueError(f"metadata['PseudoR']['{key}'] must be a scalar.")
            try:
                pr[key] = float(pr[key])
            except (TypeError, ValueError):
                raise ValueError(f"metadata['PseudoR']['{key}'] must be numeric.")

    # background offset only needed when a background mode is used
    if pr["bkg_mode"] in (0, 1):
        if pr["bkg_off"] is None:
            raise ValueError(
                "metadata['PseudoR']['bkg_off'] is required when bkg_mode is 0 or 1."
            )
        if np.ndim(pr["bkg_off"]) != 0:
            raise ValueError("metadata['PseudoR']['bkg_off'] must be a scalar.")
        try:
            pr["bkg_off"] = float(pr["bkg_off"])
        except (TypeError, ValueError):
            raise ValueError("metadata['PseudoR']['bkg_off'] must be numeric.")
    
    # -------------------------------------------------------------------------
    # start calculation
    # -------------------------------------------------------------------------
    
    # surface scattering optics
    GIXOS["fresnel"] = calc_fresnel(GIXOS["Qz"][:,pr["qxy0_select_idx"]], samp["Qc"]) 
    GIXOS["Qz_eta2"] = np.sqrt(2*2*pi*samp["tension"]/kb/samp["temperature"]/10**20)
    GIXOS["talpha_sqr"] = t_sqr(inst["alpha"], inst["energy"], qc = samp["Qc"])
    #GIXOS["Qz_array"] = np.asarray(GIXOS ["Qz"]).reshape(-1, 1) # done to convert GIXOS ["Qz"] from a row vetor to a column vector for calc_tbeta_sqr
    # Qz should always be a column vector!
    if footprint_effect and ("footprint" in inst) and ("Ddet" in inst) and ("alpha" in inst) and ("energy" in inst):
        GIXOS["dQz"] = GIXOS_dQz(GIXOS["Qz"][:,pr["qxy0_select_idx"]], inst["energy"], inst["alpha"], inst["Ddet"], inst["footprint"])     # Almost same, just not iterating through enough times(?) --> missing last row      SAME now
    else:
        GIXOS["dQz"] = np.full((len(GIXOS["tt"]), 5), np.nan)
        print("No footprint broadending calculation. For calculation: please set footprint_effect = True and provide the footprint [mm], detector distance Ddet [mm], incident angle alpha [deg] and energy [eV] in metadata field")
        
    if transmission_corr and ("Ddet" in inst) and ("alpha" in inst) and ("energy" in inst):
        if footprint_effect and ("footprint" in inst):
            GIXOS["tbeta_sqr"] = calc_tbeta_sqr(GIXOS["tt"], samp["Qc"], inst["energy"], inst["alpha"], inst["Ddet"], inst["footprint"])  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
        else:
            GIXOS["tbeta_sqr"] = calc_tbeta_sqr(GIXOS["tt"], samp["Qc"], inst["energy"], inst["alpha"], inst["Ddet"], 0.1)  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
    else:
        GIXOS["tbeta_sqr"] = np.ones((len(GIXOS["tt"]),4))
        print("no beta transmission correction. For correction: please set the transmission_corr = True and provide the detector distance Ddet [mm], incident angle alpha [deg] and energy [eV] in metadata field")
    
    if len(GIXOS["HWtt"])>1:
        DSbetaHW = GIXOS["HWtt"][pr["qxy0_select_idx"]]
        DSphiHW = GIXOS["HWtth"][0,pr["qxy0_select_idx"]]
    else:
        DSbetaHW = GIXOS["HWtt"][0]
        DSphiHW = GIXOS["HWtth"][0,0]
    
    GIXOS["r_reduced"], GIXOS["Psi_DS"], GIXOS["Psi_R"] = calc_eCWM_red_r(
                                                                            GIXOS["tt"], 
                                                                            GIXOS["tth"][0,pr["qxy0_select_idx"]], 
                                                                            alpha = inst["alpha"], 
                                                                            energy = inst["energy"], 
                                                                            DSphi_HWHM = DSphiHW, 
                                                                            DSbeta_HWHM = DSbetaHW, 
                                                                            R_resolution_mode = pr["resolution_mode"], 
                                                                            R_resolution = pr["resolution_HW"], 
                                                                            R_energy = pr["energy"], 
                                                                            R_sdd = pr["Ddet"], 
                                                                            R_bkg_mode = pr['bkg_mode'], 
                                                                            R_bkg_off = pr['bkg_off'], 
                                                                            tension = samp["tension"], 
                                                                            temp = samp["temperature"], 
                                                                            kappa = samp["kappa"], 
                                                                            amin = samp["amin"], 
                                                                            use_approx=use_approx
                                                                            )    
    
    # prefactor for this qxy0
    GIXOS['prefactor_DS'] = samp["Qc"]**4 * GIXOS["talpha_sqr"] * GIXOS["tbeta_sqr"][:, 3] / (2*GIXOS["Qz"][:,pr["qxy0_select_idx"]])**4 
    
    # computes reflectivity
    GIXOS["refl"] = np.column_stack([
        GIXOS["Qz"][:,pr["qxy0_select_idx"]],
        GIXOS["Intensity"][:,pr["qxy0_select_idx"]] / GIXOS["r_reduced"] * GIXOS["fresnel"][:, 1] / GIXOS["prefactor_DS"] / GIXOS["metadata"]["I0"],
        GIXOS["error"][:,pr["qxy0_select_idx"]] / GIXOS["r_reduced"] * GIXOS["fresnel"][:, 1] / GIXOS["prefactor_DS"] / GIXOS["metadata"]["I0"],
        GIXOS["dQz"][:, 4]
    ])
    
    # computes roughness under the resolution
    GIXOS["sigma_CW"] = np.sqrt(-np.log(GIXOS["Psi_R"])/(GIXOS["Qz"][:,pr["qxy0_select_idx"]])**2)
    
    # computes structure factor 
    GIXOS["SF"] = np.column_stack([
        GIXOS["Qz"][:,pr["qxy0_select_idx"]],
        GIXOS["Intensity"][:,pr["qxy0_select_idx"]] / GIXOS["Psi_DS"] / GIXOS["prefactor_DS"] / GIXOS["metadata"]["I0"],
        GIXOS["error"][:,pr["qxy0_select_idx"]] / GIXOS["Psi_DS"] / GIXOS["prefactor_DS"] / GIXOS["metadata"]["I0"],
        GIXOS["dQz"][:, 4]
    ])
    
    # ---- plot once ----
    if plot:
        fig_RRF, ax_RRF = GIXOS_RRF_plot(GIXOS)
        fig_R, ax_R = GIXOS_R_plot(GIXOS)
        RRFplotname = make_filename(GIXOS["metadata"], suffix="RRF.png")
        Rplotname = make_filename(GIXOS["metadata"], suffix="R.png")
        fig_RRF.savefig(RRFplotname, dpi=300, bbox_inches="tight")
        fig_R.savefig(Rplotname, dpi=300, bbox_inches="tight")
    
    
    return GIXOS # outputs GIXOS with reflectivity and structure factor added as new columns
