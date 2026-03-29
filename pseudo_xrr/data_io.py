from ruamel.yaml import YAML
import numpy as np
import matplotlib.pyplot as plt
import numbers
import pandas as pd
import math
import os
import platform
from joblib import Parallel, delayed
from scipy.integrate import dblquad
from scipy.special import kv as besselk, jv as besselj, gamma
from p08_GIXD.p08_GIXD import *
from pseudo_xrr.gixos import bulkbkg_fit, bulkbkg_plot_fit, bulkbkg_predict, GIXOS_fresnel, GIXOS_Tsqr, GIXOS_dQz, calc_eCWM_roughness_factor_DS, calc_eCWM_red_r

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


# some helping functions
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
        


#%% load data

def load_metadata(yaml_path: str):
    """
    Load metadata from a YAML file and return all parameters 
    and derived quantities matching the original script.
    """
    yaml = YAML(typ='safe')
    
    # Load YAML
    with open(yaml_path, "r") as f:
        meta = yaml.load(f)
    
    
    # linux and windows path
    if platform.system() == "Windows":
        meta["paths"]["path_xrr"] = meta["paths"]["path_xrr"].replace("/","\\")
        meta["paths"]["gixs_path"] = meta["paths"]["gixs_path"].replace("/","\\")
        meta["paths"]["path_out"] = meta["paths"]["path_out"].replace("/","\\")
    else:
        meta["paths"]["path_xrr"] = meta["paths"]["path_xrr"].replace("\\","/")
        meta["paths"]["gixs_path"] = meta["paths"]["gixs_path"].replace("\\","/")
        meta["paths"]["path_out"] = meta["paths"]["path_out"].replace("\\","/")
    
    if (meta["paths"]["path_xrr"] is None) or (meta["paths"]['xrr_datafile'] is None) or (meta["paths"]["path_xrr"].lower() == "none") or (meta["paths"]['xrr_datafile'].lower() == "none"):
        meta['xrr_data'] = None
    else: 
        meta['xrr_data'] = pd.read_csv(
            meta["paths"]["path_xrr"] + meta["paths"]['xrr_datafile'],
            delim_whitespace=True
            )
    
    # for dictionary return
    meta["measurements"]["scan"]  = np.array(meta["measurements"]["scan"], dtype=int)
    meta["measurements"]["bkgscan"] = np.array(meta["measurements"]["bkgscan"], dtype=int)
    meta["instrument"]["wavelength"] = 12404/meta["instrument"]["energy"]
    meta["qxy0"] = np.array(meta["qxy0"])
    meta['tth']= np.degrees(np.arcsin(meta["qxy0"] * meta["instrument"]["wavelength"] / 4 / np.pi)) * 2
    meta["sample_params"]["rho_b"] = meta["sample_params"]["Qc"]**2/16/math.pi
    # RFscaling exactly as in the original script
    meta['RFscaling'] = meta['measurements']['flux'] * meta['measurements']['cttime_sample'] * meta['sample_params']['rho_b'] ** 2 / math.sin(math.radians(meta['instrument']['alpha_i'])) * 4
    meta['I0'] = meta['measurements']['flux'] * meta['measurements']['cttime_sample']
        
    # # Raw parameters
    # datatype               = meta["datatype"]
    # # Paths and data loading
    # path_xrr               = meta["paths"]["path_xrr"]
    # xrr_datafile           = meta["paths"]["xrr_datafile"]
    # if (path_xrr is None) or (xrr_datafile is None) or (path_xrr.lower() == "none") or (xrr_datafile.lower() == "none"):
    #     xrr_data = None
    # else: 
    #     xrr_data = pd.read_csv(
    #         path_xrr + xrr_datafile,
    #         delim_whitespace=True
    #         )
    # path                   = meta["paths"]["path"]
    # path_out               = meta["paths"]["path_out"]
    # sample                 = meta["measurements"]["sample"]
    # scan                   = np.array(meta["measurements"]["scan"])
    # bkgsample              = meta["measurements"]["bkgsample"]
    # bkgscan                = np.array(meta["measurements"]["bkgscan"])
    # flux                   = meta["measurements"]["flux"]
    # cttime_sample          = meta["measurements"]["cttime_sample"]
    # cttime_bkg             = meta["measurements"]["cttime_bkg"]
    
    # # instrument related
    # energy                 = meta["instrument"]["energy"]
    # alpha_i                = meta["instrument"]["alpha_i"]
    # Ddet                   = meta["instrument"]["Ddet"]
    # pixel                  = meta["instrument"]["pixel"]
    # footprint              = meta["instrument"]["footprint"]
    # wavelength             = 12404 / energy

    # # 
    # qxy0                   = np.array(meta["qxy0"])
    # tth                    = np.degrees(np.arcsin(qxy0 * wavelength / 4 / np.pi)) * 2          # try as a list
    # qxy_bkg                = meta["qxy_bkg"]
    # DSpxHW                 = meta["DSpxHW"]
    # #DStthFW_px             = meta["DStthFW_px"]
    # #tth_roiHW_real         = DStthFW_px * DSpxHW
    # #DSqxyHW_real           = np.radians(tth_roiHW_real) / 2 * 4 * np.pi / wavelength * np.cos(np.radians(tth/2))
    
    # qxy0_select_idx        = meta["PseudoR"]["qxy0_select_idx"]
    # RqxyHW                 = meta["PseudoR"]["RqxyHW"]
    # #DSresHW                = meta["DSresHW"]
    # #DSqxyHW                = 2 * DSresHW
    
    # # Physical constants
    # Qc                     = meta["sample_params"]["Qc"]
    # rho_b                  = Qc**2/16/math.pi
    # kb                     = meta["sample_params"]["kb"]
    # tension                = meta["sample_params"]["tension"]
    # temperature            = meta["sample_params"]["temperature"]
    # kappa                  = meta["sample_params"]["kappa"]
    # Lk                     = math.sqrt(kappa * kb * temperature / tension) * 1e10
    # amin                   = meta["sample_params"]["amin"]
    # qmax                   = math.pi / amin
    
    # # Dependency specific parameters

    # qz_selected            = np.array(meta["dependency"]["qz_selected"])
    # kappa_deviation        = meta["dependency"]["kappa_deviation"]
    # assume_kappa           = np.array([kappa - kappa_deviation, kappa + kappa_deviation])
    
    # RFscaling exactly as in the original script
    # RFscaling = flux * cttime_sample * (math.pi / 180) ** 2 * rho_b ** 2 / math.sin(math.radians(alpha_i)) * 4
    
    # Return all variables in a dictionary
    # return {
    #     "Qc": Qc,
    #     "energy": energy,
    #     "alpha_i": alpha_i,
    #     "Ddet": Ddet,
    #     "pixel": pixel,
    #     "footprint": footprint,
    #     "wavelength": wavelength,
    #     "qxy0": qxy0,
    #     "qxy0_select_idx": qxy0_select_idx,
    #     "qxy_bkg": qxy_bkg,
    #     "RqxyHW": RqxyHW,
    #     #"DSresHW": DSresHW,
    #     #"DStthFW_px": DStthFW_px,
    #     "DSpxHW": DSpxHW,
    #     "tth": tth,
    #     #"tth_roiHW_real": tth_roiHW_real,
    #     #"DSqxyHW_real": DSqxyHW_real,
    #     "datatype": datatype,
    #     "path_xrr": path_xrr,
    #     "xrr_datafile": xrr_datafile,
    #     "xrr_data": xrr_data,
    #     "path": path,
    #     "path_out": path_out,
    #     "sample": sample,
    #     "scan": scan,
    #     "bkgsample": bkgsample,
    #     "bkgscan": bkgscan,
    #     "kb": kb,
    #     "tension": tension,
    #     "kappa": kappa,
    #     "temperature": temperature,
    #     "Lk": Lk,
    #     "amin": amin,
    #     "qmax": qmax,
    #     "RFscaling": RFscaling,
    #     #"DSqxyHW": DSqxyHW,
    #     "qz_selected": qz_selected,
    #     "kappa_deviation": kappa_deviation,
    #     "assume_kappa": assume_kappa,
    #     "rho_b": rho_b
    # }
    return meta

def load_gixos_from_meta(yaml_path: str):
    """
    Load GIXOS sample and background data according to metadata in a YAML file.
    Returns two dicts: importGIXOSdata and importbkg, each containing at least:
      - "Intensity": 2D array (rows x len(qxy0))
      - "error": 2D array
      - "tt":  1D array (mean of tt_qxy0)
      - "tth": 1D array
    for 1D data, it generate "tt_qxy0": 2D array, and calculate tt from it
    """
    # Load metadata
    meta = load_metadata(yaml_path)
    
    # Extract parameters
    datatype = meta["datatype"]
    path        = meta["paths"]["gixs_path"]
    sample      = meta["measurements"]["sample"]
    bkgsample   = meta["measurements"]["bkgsample"]
    scan        = meta["measurements"]["scan"]
    bkgscan     = meta["measurements"]["bkgscan"]
    qxy0        = meta["qxy0"]
    tth         = meta["tth"]

    importGIXOSdata = None
    importbkg       = None
    print("indicator")
    
    importGIXOSdata = load_data(f"{sample}_{scan:05d}_angle", path, metadata = yaml_path, datatype=datatype)
    importbkg = load_data(f"{bkgsample}_{bkgscan:05d}_angle", path, metadata = yaml_path, datatype=datatype)
    if "2d gixs" in datatype.lower():
        print("load 2d gixs and generate sets of 1d gixos from qxy0 list in meta, using DSpxHW binning")
        if meta["geometrical_correction"] and importGIXOSdata["HWtth"].shape == (1,1) and importGIXOSdata["HWtt"].shape == (1,):
            print("geometrical correction")
            importGIXOSdata = geometrical_corr(importGIXOSdata, Ddet = meta["instrument"]["Ddet"], det_px = meta["instrument"]["pixel"], HWtth = importGIXOSdata["HWtth"][0,0], HWtt = importGIXOSdata["HWtt"][0])
            importbkg = geometrical_corr(importbkg, Ddet = meta["instrument"]["Ddet"], det_px = meta["instrument"]["pixel"], HWtth = importGIXOSdata["HWtth"][0,0], HWtt = importGIXOSdata["HWtt"][0])
        else:
            print("geometrical correction cannot be performed when HWtth and HWtt are not unified across data")
        print("extract 1d gixos curve for qxy0 list, each binning %d pixels" %(2*meta["DSpxHW"]))
        importGIXOSdata = extract_1dGIXOS(importGIXOSdata, tth, HWpx_h = meta["DSpxHW"])
        importbkg = extract_1dGIXOS(importbkg, tth, HWpx_h = meta["DSpxHW"])
    elif "1d gixos" in datatype.lower():
        print("load 1d cut")
        print("to be implemented")
    
    # elif "1d gixos" in datatype.lower():
    #     print("load 1d cut")
    #     # Loop over each qxy0 index
    #     for idx in range(len(qxy0)):
    #         # Construct filenames
    #         fileprefix       = f"{sample}-id{scan[idx]}"
    #         GIXOSfilename    = f"{path}{fileprefix}.txt"
    #         importGIXOS_qxy0 = np.loadtxt(GIXOSfilename, skiprows=16)

    #         # Initialize storage dicts on first iteration
    #         if importGIXOSdata is None:
    #             nrows = importGIXOS_qxy0.shape[0]
    #             ncols = len(qxy0)
    #             importGIXOSdata = {
    #                 "Intensity": np.zeros((nrows, ncols)),
    #                 "tt_qxy0":   np.zeros((nrows, ncols)),
    #                 "error":     np.zeros((nrows, ncols)),
    #                 "tt":        None,
    #                 "metadata":  meta
    #             }
    #         # Fix bad pixel row 269 by averaging rows 268 & 270
    #         mean_row = np.mean(importGIXOS_qxy0[[268, 270], :], axis=0)
    #         importGIXOS_qxy0[269, :] = mean_row

    #         # Populate sample data
    #         importGIXOSdata["Intensity"][:, idx] = importGIXOS_qxy0[:, 2]
    #         importGIXOSdata["tt_qxy0"][:, idx]   = importGIXOS_qxy0[:, 1] - 0.01
    #         importGIXOSdata["error"][:, idx]     = np.sqrt(importGIXOS_qxy0[:, 2])

    #         # Background file
    #         bkgprefix        = f"{bkgsample}-id{bkgscan[idx]}"
    #         bkgfilename      = f"{path}{bkgprefix}.txt"
    #         importbkg_qxy0   = np.loadtxt(bkgfilename, skiprows=16)

    #         if importbkg is None:
    #             nrows_bkg = importbkg_qxy0.shape[0]
    #             importbkg = {
    #                 "Intensity": np.zeros((nrows_bkg, ncols)),  # should ncols be defined in here since if importGIXOSdata has values then it will not be defined in this if statment?
    #                 "tt_qxy0":   np.zeros((nrows_bkg, ncols)),
    #                 "error":     np.zeros((nrows_bkg, ncols)),
    #                 "tt":        None,
    #                 "metadata":  meta
    #             }
    #         # Fix bad pixel
    #         importbkg_qxy0[269, :] = np.mean(importbkg_qxy0[[268, 270], :], axis=0)

    #         importbkg["Intensity"][:, idx] = importbkg_qxy0[:, 2]
    #         importbkg["tt_qxy0"][:, idx]   = importGIXOS_qxy0[:, 1]
    #         importbkg["error"][:, idx]     = np.sqrt(importbkg_qxy0[:, 2])
    #         print(f"{qxy0[idx]:f}", end="\t")

    #     # Compute mean tt over qxy0 for both dicts
    #     importGIXOSdata["tt"] = np.mean(importGIXOSdata["tt_qxy0"], axis=1)
    #     importbkg["tt"]       = np.mean(importbkg["tt_qxy0"], axis=1)
    # else:
    #     print("only 2d gixs or 1d gixos cut is supported")
    return importGIXOSdata, importbkg

# Example usage:
# importGIXOSdata, importbkg = load_data_from_meta("metadata.yaml")

def load_data(gixosdataprefix, path, metadata = None, datatype = "2d gixs"):
    """
    this loads the data directly through data and path
    2d gixs data file: <prefix>_I.dat, <prefix>_tt.dat, <prefix>_tth.dat
    1d gixos cut file: <prefix>.txt
    Parameters
    ----------
    gixosdataprefix : str
        file name prefix (see above).
    path : str
        file directory, ends with /.
    metadata : str, optional
        yaml metadata file path. The default is None.
    datatype : str, optional
        either "1d gixos" or "2d gixs". The default is "2d gixs".

    Returns
    -------
    importeddata : dictionary
        field: 
            'Intensity':    intensity map, 2d or one line cut,
            ('error':       only for 1d gixos at this stage, error column)
            'tth':          tth axis in deg, 
            'tt':           tt axis in deg, 
            'HWtth':        half width of each column in tth in deg, 
            'HWtt':         half width of each row in tt in deg, 
            'HWpx_h' = .5:  horizontal half width in pixel, 
            'HWpx_v' = .5:  vertical half width in pixel.
            'metadata':     metadata, if no entry it will be None

    """
    importeddata = None
    if "2d gixs" in datatype.lower():
        '''
        currently implemented for P08 GIXS file _angle, ascii format
        by p08_GIXD.read_2D a dictionary of 'mat', 'tth', 'tt' is created
        '''
        print("load 2d image")
        importeddata = read_2D(gixosdataprefix, path, axis = ['tth', 'tt'])
        importeddata['Intensity'] = importeddata.pop('mat')
        #importeddata["HWtth"] = np.mean(np.diff(importeddata["tth"], axis = 1)/2, axis = 0, keepdims = True)
        importeddata["HWtth"] = mean1d_if_within_percent(np.diff(importeddata["tth"], axis = 1)/2)
        importeddata["HWpx_h"] = .5
        #importeddata["HWtt"] = np.mean(np.diff(importeddata["tt"], axis = 0)/2, axis = 0, keepdims = True)
        importeddata["HWtt"] = mean1d_if_within_percent(np.diff(importeddata["tt"], axis = 0)/2)
        importeddata["HWpx_v"] = .5
    elif "1d gixos" in datatype.lower():
        '''
        currently implemented for the OPLS GIXOS cut
        four columns: idx, tt(beta), intensity, qz
        '''
        print("load 1d cut")
        # Construct filenames
        GIXOSfilename    = f"{path}{gixosdataprefix}.txt"
        dataread = np.loadtxt(GIXOSfilename, skiprows=16)
        importeddata = {
            "Intensity": dataread[:, 2],
            "error":     np.sqrt(dataread[:, 2]),
            "tt":        dataread[:, 1],
            "tth":       [[]],
            "HWtth":    [],
            "HWpx_h":   .5,
            "HWtt":     np.mean(np.diff(importeddata["tt"], axis = 0)/2, axis = 0, keepdims = True),
            "HWpx_v":   .5
        }
    else:
        print("only 2d gixs or 1d gixos cut is supported")
    
    if metadata is not None:
        try:
            importeddata["metadata"] = load_metadata(metadata)
        except:
            importeddata["metadata"] = None
            print("cannot find metadata")
    else:
        importeddata["metadata"] = None
        print("metadata = None")
    
    if importeddata is None:
        print("no data is loaded")
    else:
        print('%s is loaded'% datatype)
    return importeddata


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
            'metadata':     ['instrument'] with 'energy' (eV) and 'alpha_i' (deg)
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
    if (inputdata["metadata"] is None) or ("instrument" not in inputdata["metadata"]) or (inputdata["metadata"]["instrument"] is None) or (not check_keys_numeric(["energy", "alpha_i"], inputdata["metadata"]["instrument"])):
        print("please provide energy [eV] and incident angle (alpha_i) [deg] in the ['metadata']['instrument']")
        return
        
    inputdata['mat'] = inputdata.pop('Intensity')    
    # calculate qxy, qz, and q, use the th2q function from p08_GIXD, it requires the intensity to be called mat
    outputdata = th2q(inputdata, energy = inputdata["metadata"]["instrument"]["energy"], alpha_i = inputdata["metadata"]["instrument"]["alpha_i"], absQxy = False)
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


def GIXOS_background_corr(sampledata, chamberbkg, bulkbkg_mode = None, bulkbkg_offset_lb = 0.9):
    """
    background correction, include
    - chamber background subtraction
    - wide angle (bulk) scattering background subtraction

    Parameters
    ----------
    sampledata : dictionary
        minimal field: Intensity, error, tt, tth, metadata
    chamberbkg : dictionary
        minimal field: Intensity, error, tt, tth, metadata
        same shape as sampledata
    bulkbkg_mode : string, optional
        can be None, direct, constant, fit_q. The default is None.
        None: no wide angle bkg subtraction
        direct: using the direct wide angle line for substraction
        constant: using a constant value for substraction
        fit: fitting the wide angle over Q to be used as a wide angle bkg, requires alpha_i and wavelength in metadata of the sample
            requires "Q" field in sampledata and chamberbkg
    bulkbkg_offset_lb: float, optional
        lower boundary for fitting the offset of the bulkbkg, as a factor to the average of the first 10 values of the bulkbkg GIXOS cut
        default: 0.9
    required metadata
        chamber background subtraction: ['flux'], ['cttime']. Otherwise the sample and chamber will be considered to have same integrated flux
        bulkbkg subtraction: 
            - if wide angle is used (direct or fit), ['qxy_bkg'] must exist
    
    Returns
    -------
    correcteddata : TYPE
        DESCRIPTION.

    """
    correcteddata = {   
                        "Intensity": None
                    }
    bulkbkg = {   
                        "Intensity": None
                    }
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
    
    # step 1: subtract chamber bkg
    data_chamber_subtracted = sampledata["Intensity"] - chamberbkg["Intensity"]*I0_sample_bkg
    err_propogate = np.sqrt(sampledata["error"]**2 + chamberbkg["error"]**2 * I0_sample_bkg**2)
    
    # step 2, different mode of bulk bkg subtraction
    if bulkbkg_mode is None:
        # no bulk subtraction
        print("no bulkbkg subtraction")
        correcteddata["Intensity"] = data_chamber_subtracted
        correcteddata["error"] = err_propogate
    elif isinstance(bulkbkg_mode, numbers.Number):
        # if given a number, subtract a constant
        print("cosntant bulk bkg")
        correcteddata["Intensity"] = data_chamber_subtracted - bulkbkg_mode
        correcteddata["error"] = err_propogate
        bulkbkg["Intensity"] = bulkbkg_mode
        bulkbkg["Intensity_at_GIXOS"] = bulkbkg_mode*np.ones((correcteddata["Intensity"].shape[0],correcteddata["Intensity"].shape[1]))
        correcteddata["bulkbkg"] = bulkbkg
    elif bulkbkg_mode in ["direct", "fit"]:
        # if wide angle data exist, subtract the wide angle, either directly using line cut, or using fit over q
        qxy0_idx_arr = np.where(sampledata["metadata"]["qxy0"] > sampledata["metadata"]["qxy_bkg"])[0]  # get the array of the wide angle column
        if len(qxy0_idx_arr) == 0:
            print("bkg qxy0 is smaller than the largest qxy0 position. No bulk bkg subtraction")
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
            if bulkbkg_mode == "direct":
                # directly subtract wide angle linecut
                bulkbkg["Intensity_at_GIXOS"] = np.outer(bulkbkg["Intensity"],np.ones((1,bulk_qxy0_idx[0])))
                correcteddata["Intensity"] = data_chamber_subtracted[:,:bulk_qxy0_idx[0]] - bulkbkg["Intensity_at_GIXOS"]
                correcteddata["error"] = np.sqrt(err_propogate[:,:bulk_qxy0_idx[0]]**2 + (np.outer(bulkbkg["error"],np.ones((1,bulk_qxy0_idx[0]))))**2)
                correcteddata["bulkbkg"] = bulkbkg
            else:
                # subtract the fit
                if "Q" in correcteddata:
                    bulkbkg_Q = np.mean(np.atleast_2d(correcteddata["Q"][:, bulk_qxy0_idx]), axis = 1) # this is the q axis
                    # exclude the lowest 10% of the Q range
                    Q_cut = np.min(bulkbkg_Q) + 0.1 * (np.max(bulkbkg_Q) - np.min(bulkbkg_Q))
                    mask = bulkbkg_Q >= Q_cut
                    # fit Q
                    res = bulkbkg_fit(bulkbkg_Q[mask], bulkbkg["Intensity"][mask], y0_bounds=(np.mean(bulkbkg["Intensity"][mask][0:10],axis=0)*bulkbkg_offset_lb, np.inf))   # or (200, np.inf) if you want
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
    else:
        print("please give the bulk mode among constant number, 'direct' or 'fit', or ignore it for no bulk background subtraction")
        
    
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
    Extract Qxy dependence at selected qz values and compare with CWM / eCWM.
    If fit_kappa=True, fit one global kappa to all selected rows simultaneously
    in log10-log10 space. Row-wise normalization is done at normalize_point_index.
    
    Parameters
    ----------
    GIXOSdict : TYPE
        DESCRIPTION.
    qz_targets : TYPE
        DESCRIPTION.
    * : TYPE
        DESCRIPTION.
    row_window : TYPE, optional
        DESCRIPTION. The default is 1.
    offset_factor : TYPE, optional
        DESCRIPTION. The default is 100.0.
    normalize_point_index : TYPE, optional
        DESCRIPTION. The default is 2.
    return_plot_data_only : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    results : TYPE
        DESCRIPTION.
        
    """

    # ---- parameters ----
    temperature = GIXOSdict['metadata']['sample_params']['temperature']
    tension = GIXOSdict['metadata']['sample_params']['tension']
    amin = GIXOSdict['metadata']['sample_params']['amin']
    kappa = GIXOSdict['metadata']['sample_params']['kappa']

    energy = GIXOSdict['metadata']['instrument']['energy']
    alpha_i = GIXOSdict['metadata']['instrument']['alpha_i']
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
        # calc_eCWM_roughness_factor_DS(alpha, beta_space, phi, energy, DSphi_HWHM, DSbeta_HWHM,
        #                           tension, temp, kappa, amin, use_approx=False):
        for phi_value in phi:
            # _, ds_model, _ = calc_film_DS_RRF_integ(
            #     beta_space[row_index],
            #     phi_value,
            #     energy,
            #     alpha_i,
            #     2e-4,
            #     HWtth,
            #     HWtt * (row_window * 2 + 1),
            #     tension,
            #     temperature,
            #     kappa_value,
            #     amin,
            #     use_approx=True,
            #     show_plot=False
            # )
            
            ds_model = calc_eCWM_roughness_factor_DS(
                alpha_i,
                beta_space[row_index],
                phi_value,
                energy,
                HWtth,
                HWtt * (row_window * 2 + 1),
                tension,
                temperature,
                kappa_value,
                amin,
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
            GIXOS_qxy_dependence_plot(
                results,
                title = rf"$Q_{{xy}}$ dependence fit, $\kappa = {kappa_use:.0f}\pm{kappa_err:.0f}\,k_{{\mathrm{{B}}}}T$"
            )
        else:
            GIXOS_qxy_dependence_plot(
                results,
                title = rf"$Q_{{xy}}$ dependence predict, $\kappa = {kappa_use:.0f}\,k_{{B}}T$"
            )

    return results

def GIXOS_qxy_dependence_plot(results, *, show_refs=True, show_err = True, title=None):
    """
    Optional plotting helper (expects matplotlib imported as plt in your environment).
    Plots offset linecuts in log-log, and overlays two reference curves.
    """
    plt.figure(figsize=(5, 7))
    for idx, qxy in enumerate(results['Qxy']):

        plt.plot(qxy, results['I_sum_offset'][idx,:], marker="o", linestyle="None",
                 label=f"$Q_{{z}}≈{results['target_qz'][idx]:.2f}\AA^{{-1}}$")

        if show_refs and results["ref_CWM"] is not None:
            plt.plot(qxy, results["ref_CWM"][idx,:], linestyle=":", color="k", linewidth=1.4)
        if show_refs and results["ref_CWM"]  is not None:
            plt.plot(qxy, results["ref_eCWM"][idx,:], linestyle="-", color="k", linewidth=1.8)
        if show_err and "ref_eCWM_err_u" in results and "ref_eCWM_err_l" in results:
            plt.plot(qxy, results["ref_eCWM_err_u"][idx,:], linestyle="--", color="k", linewidth=1.4)
            plt.plot(qxy, results["ref_eCWM_err_l"][idx,:], linestyle="--", color="k", linewidth=1.4)
        
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'$Q_{xy}\/(\AA^{-1})$')
    plt.ylabel("(GI) diffuse intensity (offset)")
    if title is not None:
        plt.title(title)
    else:
        plt.title(f"$Q_{{xy}}$ dependence")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    plt.show()


# processing into SF and RRF
def eCWM_analysis_old(GIXOS, transmission_corr = False, footprint_effect = False, use_approx = False):
    """
    name changed to eCWM analysis
    use metadata imbedded in the input data, no DSbetaHW
    input for calc_film_DS_RRF_integ: use DSphi_HW instead of DSqxy_HW, use tth instead of qxy0, use energy in eV instead of keV
    
    """
    GIXOS["fresnel"] = GIXOS_fresnel(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["sample_params"]["Qc"]) # check if fresnel == GIXOS_fresnel      SAME
    #GIXOS["Qz_array"] = np.asarray(GIXOS ["Qz"]).reshape(-1, 1) # done to convert GIXOS ["Qz"] from a row vetor to a column vector for GIXOS_Tsqr
    # Qz should always be a column vector!
    if footprint_effect and ("footprint" in GIXOS["metadata"]["instrument"]) and ("Ddet" in GIXOS["metadata"]["instrument"]) and ("alpha_i" in GIXOS["metadata"]["instrument"]) and ("energy" in GIXOS["metadata"]["instrument"]):
        GIXOS["dQz"] = GIXOS_dQz(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["instrument"]["Ddet"], GIXOS["metadata"]["instrument"]["footprint"])     # Almost same, just not iterating through enough times(?) --> missing last row      SAME now
    else:
        GIXOS["dQz"] = np.ones((len(GIXOS["tt"]),5))
        print("No footprint broadending calculation. For calculation: please set footprint_effect = True and provide the footprint [mm], detector distance Ddet [mm], incident angle alpha_i [deg] and energy [eV] in metadata field")
        
    if transmission_corr and ("Ddet" in GIXOS["metadata"]["instrument"]) and ("alpha_i" in GIXOS["metadata"]["instrument"]) and ("energy" in GIXOS["metadata"]["instrument"]):
        if footprint_effect and ("footprint" in GIXOS["metadata"]["instrument"]):
            GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["sample_params"]["Qc"], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["instrument"]["Ddet"], GIXOS["metadata"]["instrument"]["footprint"])  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
        else:
            GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["sample_params"]["Qc"], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["instrument"]["Ddet"], 0.1)  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
    else:
        GIXOS["transmission"] = np.ones((len(GIXOS["tt"]),4))
        print("no transmission correction. For correction: please set the transmission_corr = True and provide the detector distance Ddet [mm], incident angle alpha_i [deg] and energy [eV] in metadata field")
    
    if len(GIXOS["HWtt"])>1:
        DSbetaHW = GIXOS["HWtt"][GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]]
        DSphiHW = GIXOS["HWtth"][0,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]]
    else:
        DSbetaHW = GIXOS["HWtt"][0]
        DSphiHW = GIXOS["HWtth"][0,0]
    GIXOS["DS_RRF_integ"], GIXOS["DS_term_integ"], GIXOS["RRF_term_integ"] = calc_film_DS_RRF_integ(GIXOS["tt"], GIXOS["tth"][0,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["PseudoR"]["RqxyHW"], DSphiHW, DSbetaHW, GIXOS["metadata"]["sample_params"]["tension"], GIXOS["metadata"]["sample_params"]["temperature"], GIXOS["metadata"]["sample_params"]["kappa"], GIXOS["metadata"]["sample_params"]["amin"], use_approx=use_approx)
    # DS = Diffuse Scatter; RRF = Specular Reflectivity Normalized by Fresnel Reflectivity
    # Approx form is derived from Taylor expansion, which is dependent on being close to 0 angle --> higher deviations at high angles

    # computes reflectivity
    GIXOS["refl"] = np.column_stack([
        GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]],
        GIXOS["Intensity"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["transmission"][:, 3],
        GIXOS["error"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["transmission"][:, 3],
        GIXOS["dQz"][:, 4]
    ])

    # computes structure factor 
    GIXOS["SF"] = np.column_stack([
        GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]],
        GIXOS["Intensity"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_term_integ"] / GIXOS["transmission"][:, 3],
        GIXOS["error"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_term_integ"] / GIXOS["transmission"][:, 3],
        GIXOS["dQz"][:, 4]
    ])
    return GIXOS # outputs GIXOS with reflectivity and structure factor added as new columns


# processing into SF and RRF
def GIXOS2R(GIXOS, transmission_corr = False, footprint_effect = False, use_approx = False):
    """
    name changed to GIXOS2R
    use metadata imbedded in the input data, no DSbetaHW
    input for calc_film_DS_RRF_integ: use DSphi_HW instead of DSqxy_HW, use tth instead of qxy0, use energy in eV instead of keV
    
    """
    GIXOS["fresnel"] = GIXOS_fresnel(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["sample_params"]["Qc"]) # check if fresnel == GIXOS_fresnel      SAME
    #GIXOS["Qz_array"] = np.asarray(GIXOS ["Qz"]).reshape(-1, 1) # done to convert GIXOS ["Qz"] from a row vetor to a column vector for GIXOS_Tsqr
    # Qz should always be a column vector!
    if footprint_effect and ("footprint" in GIXOS["metadata"]["instrument"]) and ("Ddet" in GIXOS["metadata"]["instrument"]) and ("alpha_i" in GIXOS["metadata"]["instrument"]) and ("energy" in GIXOS["metadata"]["instrument"]):
        GIXOS["dQz"] = GIXOS_dQz(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["instrument"]["Ddet"], GIXOS["metadata"]["instrument"]["footprint"])     # Almost same, just not iterating through enough times(?) --> missing last row      SAME now
    else:
        GIXOS["dQz"] = np.ones((len(GIXOS["tt"]),5))
        print("No footprint broadending calculation. For calculation: please set footprint_effect = True and provide the footprint [mm], detector distance Ddet [mm], incident angle alpha_i [deg] and energy [eV] in metadata field")
        
    if transmission_corr and ("Ddet" in GIXOS["metadata"]["instrument"]) and ("alpha_i" in GIXOS["metadata"]["instrument"]) and ("energy" in GIXOS["metadata"]["instrument"]):
        if footprint_effect and ("footprint" in GIXOS["metadata"]["instrument"]):
            GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["sample_params"]["Qc"], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["instrument"]["Ddet"], GIXOS["metadata"]["instrument"]["footprint"])  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
        else:
            GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["sample_params"]["Qc"], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["instrument"]["Ddet"], 0.1)  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
    else:
        GIXOS["transmission"] = np.ones((len(GIXOS["tt"]),4))
        print("no transmission correction. For correction: please set the transmission_corr = True and provide the detector distance Ddet [mm], incident angle alpha_i [deg] and energy [eV] in metadata field")
    
    if len(GIXOS["HWtt"])>1:
        DSbetaHW = GIXOS["HWtt"][GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]]
        DSphiHW = GIXOS["HWtth"][0,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]]
    else:
        DSbetaHW = GIXOS["HWtt"][0]
        DSphiHW = GIXOS["HWtth"][0,0]
    
    GIXOS["DS_RRF_integ"], GIXOS["DS_term_integ"], GIXOS["RRF_term_integ"] = calc_eCWM_red_r(GIXOS["tt"], GIXOS["tth"][0,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]], GIXOS["metadata"]["instrument"]["energy"], GIXOS["metadata"]["instrument"]["alpha_i"], GIXOS["metadata"]["PseudoR"]["RqxyHW"], DSphiHW, DSbetaHW, GIXOS["metadata"]["sample_params"]["tension"], GIXOS["metadata"]["sample_params"]["temperature"], GIXOS["metadata"]["sample_params"]["kappa"], GIXOS["metadata"]["sample_params"]["amin"], use_approx=use_approx)
    # DS = Diffuse Scatter; RRF = Specular Reflectivity Normalized by Fresnel Reflectivity
    # Approx form is derived from Taylor expansion, which is dependent on being close to 0 angle --> higher deviations at high angles
    
    # prefactor for this qxy0
    GIXOS['prefactor_DS'] = GIXOS["metadata"]["sample_params"]["Qc"]**4 * 4 * GIXOS["transmission"][:, 3] / (2*GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]])**4 
    
    # computes reflectivity
    GIXOS["refl"] = np.column_stack([
        GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]],
        GIXOS["Intensity"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["prefactor_DS"],
        GIXOS["error"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["prefactor_DS"],
        GIXOS["dQz"][:, 4]
    ])

    # computes structure factor 
    GIXOS["SF"] = np.column_stack([
        GIXOS["Qz"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]],
        GIXOS["Intensity"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_term_integ"] / GIXOS["prefactor_DS"],
        GIXOS["error"][:,GIXOS["metadata"]["PseudoR"]["qxy0_select_idx"]] / GIXOS["DS_term_integ"] / GIXOS["prefactor_DS"],
        GIXOS["dQz"][:, 4]
    ])
    return GIXOS # outputs GIXOS with reflectivity and structure factor added as new columns


def GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step):
    xrrfilename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_R_PYTHON_TEST.dat" # becomes "instrument_46392_R_PYTHON.dat" - qz and dqz columns are very accurate, but R and dR start off semi-accurate but increasingly deviate after ~15th value
    with open(xrrfilename, 'w') as f:
        f.write(f"# files\n")
        f.write(f"sample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgsample']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha_i']}\n")
        f.write(f"footprint [mm]: {metadata['footprint']:.1f}\n")
        f.write(f"sdd [mm]: {metadata['Ddet']:.2f}\n")
        f.write(f"qxy resolution HWHM at specular [A^-1]: {metadata['DSresHW']}\n")
        f.write(f"phi_opening [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"beta_step [deg]: {tt_step}\n")
        f.write(f"# DS-XRR conversion optics setting\n")
        f.write(f"phi [deg]: {metadata['tth']}\n")
        f.write(f"qxy(beta=0) [A^-1]: {metadata['qxy0'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"phi integration HW [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"corresponding qxy HW [A^-1]: {metadata['DSqxyHW_real']}\n")
        f.write(f"R slit: {xrr_config['slit_v']} mm (v) {xrr_config['slit_h']} mm (h) at {xrr_config['sdd']} mm distance, {xrr_config['energy']} eV beam energy\n")
        f.write(f"scaling: {metadata['RFscaling']}\n")
        f.write(f"# DS-XRR conversion sample setting\n")
        f.write(f"tension [N/m]: {metadata['tension']}\n")
        f.write(f"temperature [K]: {metadata['temperature']:.1f}\n")
        f.write(f"kappa [kbT]: {metadata['kappa']:.1f}\n")
        f.write(f"CW short cutoff [A]: {metadata['amin']}\n")
        f.write(f"CW and Kapa roughness [A]: {GIXOS['refl_roughness'][0]} to {GIXOS['refl_roughness'][-1]}\n")
        f.write("# data\nqz\tR\tdR\tdqz\n[A^-1]\t[a.u.]\t[a.u.]\t[A^-1]\n")

    # Save reflectivity data
    with open(xrrfilename, 'a') as f:
        np.savetxt(f, GIXOS["refl_recSlit"], delimiter='\t', fmt='%.6e')
    #    np.savetxt(f, refl_recSlit, delimiter='\t', fmt='%.6e', comments='', header='', encoding='utf-8', newline='\n', append=True)


    # ---- FILE 2: DS/(R/RF) ----
    ds2rrf_filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_DS2RRF_PYTHON_TEST.dat" # becomes "instrument_46392_DS2RRF_PYTHON.dat" - first column, less than 0.1% error ; if we approx at the same decimal point that MATLAB appears to round off at, would be the same values - second column: starts semi close (less than 0.1% error), but deviates heavily by the end (~x2.5 the actual value it is supposed to have)
    with open(ds2rrf_filename, 'w') as f:
        f.write(f"# files\n")
        f.write(f"sample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgsample']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha_i']}\n")
        f.write(f"footprint [mm]: {metadata['footprint']:.1f}\n")
        f.write(f"sdd [mm]: {metadata['Ddet']:.2f}\n")
        f.write(f"qxy resolution HWHM at specular [A^-1]: {metadata['DSresHW']}\n")
        f.write(f"phi_opening [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"beta_step [deg]: {tt_step}\n")
        f.write(f"# DS-XRR conversion optics setting\n")
        f.write(f"phi [deg]: {metadata['tth']}\n")   
        f.write(f"qxy(beta=0) [A^-1]: {metadata['qxy0'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"phi integration HW [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"corresponding qxy HW [A^-1]: {metadata['DSqxyHW_real']}\n")
        f.write(f"R slit: {xrr_config['slit_v']} mm (v) {xrr_config['slit_h']} mm (h) at {xrr_config['sdd']} mm distance, {xrr_config['energy']} eV beam energy\n")
        f.write(f"scaling: {metadata['RFscaling']}\n")
        f.write(f"# DS-XRR conversion sample setting\n")
        f.write(f"tension [N/m]: {metadata['tension']}\n")
        f.write(f"temperature [K]: {metadata['temperature']:.1f}\n")
        f.write(f"kappa [kbT]: {metadata['kappa']:.1f}\n")
        f.write(f"CW short cutoff [A]: {metadata['amin']}\n")
        f.write(f"CW and Kapa roughness [A]: {GIXOS['refl_roughness'][0]} to {GIXOS['refl_roughness'][-1]}\n")
        f.write("# data\nqz\tDS/(R/RF)\n[A^-1]\t[a.u.]\n")

    ds_over_rrf = GIXOS["DS_term_integ"] / (xrr_config["Rterm_rect_slit"] / xrr_config["RF"])
    with open(ds2rrf_filename, 'a') as f:
        np.savetxt(f, np.column_stack((GIXOS["Qz"], ds_over_rrf)), delimiter='\t', fmt='%.6e')
    #     np.savetxt(f, np.column_stack((GIXOS["Qz"], ds_over_rrf)), delimiter='\t', fmt='%.6e', comments='', header='', encoding='utf-8', newline='\n', append=True)


    # ---- FILE 3: Structure Factor ----
    sf_filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_SF_PYTHON_TEST.dat" # becomes "instrument_46392_SF_PYTHON.dat" - first four columns are largely accurate; last 2 columns deviate (first is off by ~8%, second is off by a larger margin but both start semi-close and then increasingly deviate as index increases)
    with open(sf_filename, 'w') as f:
        f.write(f"# pure structure factor and kapa/cw roughness with its decay term under given XRR resolution\n")
        f.write(f"# files\nsample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgscan']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")    
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha_i']}\n")
        f.write(f"footprint [mm]: {metadata['footprint']:.1f}\n")
        f.write(f"sdd [mm]: {metadata['Ddet']:.2f}\n")
        f.write(f"qxy resolution HWHM at specular [A^-1]: {metadata['DSresHW']}\n")
        f.write(f"phi_opening [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"beta_step [deg]: {tt_step}\n")
        f.write(f"# DS-XRR conversion optics setting\n")
        f.write(f"phi [deg]: {metadata['tth']}\n")   
        f.write(f"qxy(beta=0) [A^-1]: {metadata['qxy0'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"phi integration HW [deg]: {metadata['tth_roiHW_real']}\n")
        f.write(f"corresponding qxy HW [A^-1]: {metadata['DSqxyHW_real']}\n")
        f.write(f"R slit: {xrr_config['slit_v']} mm (v) {xrr_config['slit_h']} mm (h) at {xrr_config['sdd']} mm distance, {xrr_config['energy']} eV beam energy\n")
        f.write(f"scaling: {metadata['RFscaling']}\n")
        f.write(f"# DS-XRR conversion sample setting\n")
        f.write(f"tension [N/m]: {metadata['tension']}\n")
        f.write(f"temperature [K]: {metadata['temperature']:.1f}\n")
        f.write(f"kappa [kbT]: {metadata['kappa']:.1f}\n")
        f.write(f"CW short cutoff [A]: {metadata['amin']}\n")
        f.write(f"CW and Kapa roughness [A]: {GIXOS['refl_roughness'][0]} to {GIXOS['refl_roughness'][-1]}\n")
        f.write("# data\nqz\tSF\tdSF\tdQz\tsigma_R\texp(-qz2sigma2)\n[A^-1]\t[a.u.]\t[a.u.]\t[A^-1]\t[A^-1]\t[a.u.]\n")

    with open(sf_filename, 'a') as f:
        np.savetxt(f, GIXOS["SF"], delimiter='\t', fmt='%.6e')



# from pseudo_xrr.plots import GIXOS_data_plot, R_data_plot, R_pseudo_data_plot
# def rectangular_slit(metadata_file = './testing_data/gixos_metadata.yaml'):     # can make this a main function to run the whole code and have a parameter be the text file
#     importGIXOSdata, importbkg = load_data(metadata_file)
#     metadata = load_metadata(metadata_file)
#     importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
#     importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)
#     metadata = real_space_2theta(metadata)
#     GIXOS, DSbetaHW = GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
#     GIXOS_data_plot(GIXOS, metadata)
#     GIXOS = GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW)
#     xrr_config = rect_slit_function(GIXOS, metadata)
#     GIXOS = conversion_to_reflectivity(GIXOS, xrr_config)
#     print("xrr_config keys:", xrr_config.keys())

#     GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step)
#     R_data_plot(GIXOS, metadata, xrr_config)
#     R_pseudo_data_plot(GIXOS, metadata, xrr_config)





# import copy

# def create_dependency_models(GIXOS, metadata, DSbetaHW):
#     model = {
#         "tt": np.ones(len(metadata['qz_selected'])),
#         "Qz": np.ones(len(metadata['qz_selected'])),
#         "Qxy": np.zeros((len(metadata['qz_selected']), GIXOS["Qxy"].shape[1]))
#     }

#     for idx in range(len(metadata['qz_selected'])):
#         rowidx_arr = np.where(GIXOS["Qz"] <= metadata['qz_selected'][idx])[0]
#         rowidx = rowidx_arr[-1]
#         model["tt"][idx] = GIXOS["tt"][rowidx]
#         model["Qz"][idx] = GIXOS["Qz"][rowidx]
#         model["Qxy"][idx, :] = GIXOS["Qxy"][rowidx, :]

#     assume_model = {
#         "1": copy.deepcopy(model),
#         "2": copy.deepcopy(model)
#     }

#     CWM_model = copy.deepcopy(model)

#     model["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     model["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     model["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     assume_model["1"]["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     assume_model["1"]["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     assume_model["1"]["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     assume_model["2"]["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     assume_model["2"]["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     assume_model["2"]["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     CWM_model["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     CWM_model["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
#     CWM_model["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))

#     metadata["energy"] = np.asarray(metadata["energy"])
#     metadata["alpha_i"] = np.asarray(metadata["alpha_i"])
#     metadata["RqxyHW"] = np.asarray(metadata["RqxyHW"])
#     metadata["DSqxyHW_real"] = np.asarray(metadata["DSqxyHW_real"])
#     # GIXOS["DSbetaHW"] = np.asarray(GIXOS["DSbetaHW"])        only add this and make changes if we say that DSbetaHW is part of GIXOS above
#     DSbetaHW = np.asarray(DSbetaHW)
#     metadata["tension"] = np.asarray(metadata["tension"])
#     metadata["temperature"] = np.asarray(metadata["temperature"])
#     metadata["kappa"] = np.asarray(metadata["kappa"])
#     metadata["amin"] = np.asarray(metadata["amin"])


#     def process(idx):
#         show_last_plot = (idx == GIXOS["GIXOS"].shape[1] - 1)  # Only show plot on last iteration
#         model_DS_RRF, model_DS_term, model_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["tth"][idx], metadata['energy'], metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], metadata['kappa'], metadata['amin'], show_plot = False)
#         assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["tth"][idx], metadata['energy'], metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], metadata['assume_kappa'][0], metadata['amin'], show_plot = False)
#         assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["tth"][idx], metadata['energy'], metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], metadata['assume_kappa'][1], metadata['amin'], show_plot = False)
#         CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["tth"][idx], metadata['energy'], metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], 0, metadata['amin'], show_plot = show_last_plot)
#         return idx, model_DS_RRF, model_DS_term, model_RRF_term, assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term, assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term, CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term
#     results = Parallel(n_jobs=-1, backend="loky")(
#         delayed(process)(i) for i in range(GIXOS["GIXOS"].shape[1])
#     )

#     for idx, model_DS_RRF, model_DS_term, model_RRF_term, assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term, assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term, CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term in results:
#         model["DS_RRF"][:, idx], model["DS_term"][:, idx], model["RRF_term"][:, idx] = model_DS_RRF, model_DS_term, model_RRF_term
#         assume_model["1"]["DS_RRF"][:, idx], assume_model["1"]["DS_term"][:, idx], assume_model["1"]["RRF_term"][:, idx] = assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term
#         assume_model["2"]["DS_RRF"][:, idx], assume_model["2"]["DS_term"][:, idx], assume_model["2"]["RRF_term"][:, idx] = assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term
#         CWM_model["DS_RRF"][:, idx], CWM_model["DS_term"][:, idx], CWM_model["RRF_term"][:, idx] = CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term
#     return model, assume_model, CWM_model