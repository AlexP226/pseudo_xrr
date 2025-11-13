from ruamel.yaml import YAML
import numpy as np
import numbers
import pandas as pd
import math
import os
import platform
from joblib import Parallel, delayed
from scipy.integrate import dblquad
from scipy.special import kv as besselk, jv as besselj, gamma
from p08_GIXD.p08_GIXD import *
from pseudo_xrr.gixos import GIXOS_fresnel, GIXOS_Tsqr, GIXOS_dQz, calc_film_DS_RRF_integ, film_integral_delta_beta_delta_phi

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

'''
    
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
        meta["paths"]["path"] = meta["paths"]["path"].replace("/","\\")
        meta["paths"]["path_out"] = meta["paths"]["path_out"].replace("/","\\")
    else:
        meta["paths"]["path_xrr"] = meta["paths"]["path_xrr"].replace("\\","/")
        meta["paths"]["path"] = meta["paths"]["path"].replace("\\","/")
        meta["paths"]["path_out"] = meta["paths"]["path_out"].replace("\\","/")
    
    # Raw parameters
    colorset               = meta["colorset"]
    gamma_E                = meta["gamma_E"]
    Qc                     = meta["Qc"]
    energy                 = meta["energy"]
    alpha_i                = meta["alpha_i"]
    Ddet                   = meta["Ddet"]
    pixel                  = meta["pixel"]
    footprint              = meta["footprint"]
    wavelength             = 12404 / energy
    rho_b                  = Qc**2/16/math.pi
    
    qxy0                   = np.array(meta["qxy0"])
    tth                    = np.degrees(np.arcsin(qxy0 * wavelength / 4 / np.pi)) * 2          # try as a list
    qxy0_select_idx        = meta["qxy0_select_idx"]
    qxy_bkg                = meta["qxy_bkg"]

    RqxyHW                 = meta["RqxyHW"]
    DSresHW                = meta["DSresHW"]
    DStthFW_px             = meta["DStthFW_px"]
    DSpxHW                 = meta["DSpxHW"]
    tth_roiHW_real         = DStthFW_px * DSpxHW
    DSqxyHW_real           = np.radians(tth_roiHW_real) / 2 * 4 * np.pi / wavelength * np.cos(np.radians(tth/2))

    datatype               = meta["datatype"]
    # Paths and data loading
    path_xrr               = meta["paths"]["path_xrr"]
    xrr_datafile           = meta["paths"]["xrr_datafile"]
    if (path_xrr is None) or (xrr_datafile is None) or (path_xrr.lower() == "none") or (xrr_datafile.lower() == "none"):
        xrr_data = None
    else: 
        xrr_data = pd.read_csv(
            path_xrr + xrr_datafile,
            delim_whitespace=True
            )

    path                   = meta["paths"]["path"]
    path_out               = meta["paths"]["path_out"]

    sample                 = meta["sample"]
    scan                   = np.array(meta["scan"])
    bkgsample              = meta["bkgsample"]
    bkgscan                = np.array(meta["bkgscan"])
    
    flux                   = meta["flux"]
    cttime_sample          = meta["cttime_sample"]
    cttime_bkg             = meta["cttime_bkg"]
    
    # Physical constants
    kb                     = meta["physical_constants"]["kb"]
    tension                = meta["physical_constants"]["tension"]
    temperature            = meta["physical_constants"]["temperature"]

    # Derived physics quantities
    kappa                  = meta["derived"]["kappa"]
    Lk                     = math.sqrt(kappa * kb * temperature / tension) * 1e10
    amin                   = meta["derived"]["amin"]
    qmax                   = math.pi / amin
    
    # Dependency specific parameters
    DSqxyHW                = 2 * DSresHW
    qz_selected            = np.array(meta["dependency"]["qz_selected"])
    kappa_deviation        = meta["dependency"]["kappa_deviation"]
    assume_kappa           = np.array([kappa - kappa_deviation, kappa + kappa_deviation])
    
    # RFscaling exactly as in the original script
    RFscaling = flux * cttime_sample * (math.pi / 180) ** 2 * rho_b ** 2 / math.sin(math.radians(alpha_i)) * 4
    
    # Return all variables in a dictionary
    return {
        "colorset": colorset,
        "gamma_E": gamma_E,
        "Qc": Qc,
        "energy": energy,
        "alpha_i": alpha_i,
        "Ddet": Ddet,
        "pixel": pixel,
        "footprint": footprint,
        "wavelength": wavelength,
        "qxy0": qxy0,
        "qxy0_select_idx": qxy0_select_idx,
        "qxy_bkg": qxy_bkg,
        "RqxyHW": RqxyHW,
        "DSresHW": DSresHW,
        "DStthFW_px": DStthFW_px,
        "DSpxHW": DSpxHW,
        "tth": tth,
        "tth_roiHW_real": tth_roiHW_real,
        "DSqxyHW_real": DSqxyHW_real,
        "datatype": datatype,
        "path_xrr": path_xrr,
        "xrr_datafile": xrr_datafile,
        "xrr_data": xrr_data,
        "path": path,
        "path_out": path_out,
        "sample": sample,
        "scan": scan,
        "bkgsample": bkgsample,
        "bkgscan": bkgscan,
        "kb": kb,
        "tension": tension,
        "kappa": kappa,
        "temperature": temperature,
        "Lk": Lk,
        "amin": amin,
        "qmax": qmax,
        "RFscaling": RFscaling,
        "DSqxyHW": DSqxyHW,
        "qz_selected": qz_selected,
        "kappa_deviation": kappa_deviation,
        "assume_kappa": assume_kappa,
        "rho_b": rho_b
    }

def load_data_from_meta(yaml_path: str):
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
    #with open(yaml_path, "r") as f:
    #    meta = yaml.safe_load(f)
    meta = load_metadata(yaml_path)
    
    # Extract parameters
    sample      = meta["sample"]
    bkgsample   = meta["bkgsample"]
    path        = meta["path"]
    qxy0        = np.array(meta["qxy0"])
    tth         = np.array(meta["tth"])
    scan        = np.array(meta["scan"], dtype=int)
    bkgscan     = np.array(meta["bkgscan"], dtype=int)
    datatype = meta["datatype"]

    importGIXOSdata = None
    importbkg       = None
    print("indicator")
    
    if "2d gixs" in datatype.lower():
        print("load 2d image")
        import2ddata = read_2D(f"{sample}_{scan:05d}_angle", path, axis = ['tth', 'tt'])
        importGIXOSdata = extract_1dGIXOS(import2ddata, tth, DSpxHW = meta["DSpxHW"])
        import2ddata_bkg = read_2D(f"{bkgsample}_{bkgscan:05d}_angle", path, axis = ['tth', 'tt'])
        importbkg = extract_1dGIXOS(import2ddata_bkg, tth, DSpxHW = meta["DSpxHW"])
        
    elif "1d gixos" in datatype.lower():
        print("load 1d cut")
        # Loop over each qxy0 index
        for idx in range(len(qxy0)):
            # Construct filenames
            fileprefix       = f"{sample}-id{scan[idx]}"
            GIXOSfilename    = f"{path}{fileprefix}.txt"
            importGIXOS_qxy0 = np.loadtxt(GIXOSfilename, skiprows=16)

            # Initialize storage dicts on first iteration
            if importGIXOSdata is None:
                nrows = importGIXOS_qxy0.shape[0]
                ncols = len(qxy0)
                importGIXOSdata = {
                    "Intensity": np.zeros((nrows, ncols)),
                    "tt_qxy0":   np.zeros((nrows, ncols)),
                    "error":     np.zeros((nrows, ncols)),
                    "tt":        None
                }
            # Fix bad pixel row 269 by averaging rows 268 & 270
            mean_row = np.mean(importGIXOS_qxy0[[268, 270], :], axis=0)
            importGIXOS_qxy0[269, :] = mean_row

            # Populate sample data
            importGIXOSdata["Intensity"][:, idx] = importGIXOS_qxy0[:, 2]
            importGIXOSdata["tt_qxy0"][:, idx]   = importGIXOS_qxy0[:, 1] - 0.01
            importGIXOSdata["error"][:, idx]     = np.sqrt(importGIXOS_qxy0[:, 2])

            # Background file
            bkgprefix        = f"{bkgsample}-id{bkgscan[idx]}"
            bkgfilename      = f"{path}{bkgprefix}.txt"
            importbkg_qxy0   = np.loadtxt(bkgfilename, skiprows=16)

            if importbkg is None:
                nrows_bkg = importbkg_qxy0.shape[0]
                importbkg = {
                    "Intensity": np.zeros((nrows_bkg, ncols)),  # should ncols be defined in here since if importGIXOSdata has values then it will not be defined in this if statment?
                    "tt_qxy0":   np.zeros((nrows_bkg, ncols)),
                    "error":     np.zeros((nrows_bkg, ncols)),
                    "tt":        None
                }
            # Fix bad pixel
            importbkg_qxy0[269, :] = np.mean(importbkg_qxy0[[268, 270], :], axis=0)

            importbkg["Intensity"][:, idx] = importbkg_qxy0[:, 2]
            importbkg["tt_qxy0"][:, idx]   = importGIXOS_qxy0[:, 1]
            importbkg["error"][:, idx]     = np.sqrt(importbkg_qxy0[:, 2])
            print(f"{qxy0[idx]:f}", end="\t")

        # Compute mean tt over qxy0 for both dicts
        importGIXOSdata["tt"] = np.mean(importGIXOSdata["tt_qxy0"], axis=1)
        importbkg["tt"]       = np.mean(importbkg["tt_qxy0"], axis=1)
    else:
        print("only 2d gixs or 1d gixos cut is supported")
    return importGIXOSdata, importbkg

# Example usage:
# importGIXOSdata, importbkg = load_data("metadata.yaml")

def load_data(gixosdataprefix, path, metadata = None, datatype = "2d gixs"):
    '''
    this loads the data directly through data and path
    '''
    importeddata = None
    if "2d gixs" in datatype.lower():
        '''
        currently implemented for P08 GIXS file _angle, ascii format
        by p08_GIXD.read_2D a dictionary of 'mat', 'tth', 'tt' is created
        '''
        print("load 2d image")
        importeddata = read_2D(gixosdataprefix, path, axis = ['tth', 'tt'])
        importeddata['Intensity'] = importeddata.pop('mat')
        importeddata["HWtth"] = np.mean(importeddata ["tth"][:,1:] - importeddata ["tth"][:,0:-1], axis = 1)/2
        importeddata["HWpx_h"] = .5
        if importeddata["tt"].ndim>1:
            importeddata["HWtt"] = np.mean(importeddata ["tt"][1:,:] - importeddata ["tt"][0:-1,:], axis = 0)/2
        else:
            importeddata["HWtt"] = np.array(np.mean(importeddata ["tt"][1:] - importeddata ["tt"][0:-1])/2)
        importeddata["HWpx_v"] = .5
    elif "1d gixos" in datatype.lower():
        '''
        currently implemented for the OPLS GIXOS cut
        '''
        print("load 1d cut")
        # Construct filenames
        GIXOSfilename    = f"{path}{fileprefix}.txt"
        dataread = np.loadtxt(GIXOSfilename, skiprows=16)
        importeddata = {
            "Intensity": dataread[:, 2],
            "error":     dataread[:, 1],
            "tt":        np.sqrt(dataread[:, 2]),
            "tth":       [[]],
            "HWtth":    [],
            "HWpx_h":   .5,
            "HWtt":    np.array(np.mean(importeddata ["tt"][1:] - importeddata ["tt"][0:-1])/2),
            "HWpx_v":   .5
        }
    else:
        print("only 2d gixs or 1d gixos cut is supported")
    
    importeddata["metadata"] = metadata
    
    if importeddata is None:
        print("no data is loaded")
    else:
        print('%s is loaded'% datatype)
    return importeddata

def geometrical_corr(gixs2d, Ddet = 560.7, det_px = 0.075, HWtth = None, HWtt = None):
    """
    in case the geometrical correction is not done in the rebinned data (2D)
    for detector perpendicular to the surface
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
    '''
    binning GIXOS data in vertical direction
    - originally work with both sample data and bkg
    - now make the general that only work with data
    '''
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


def GIXOS_background_corr(sampledata, chamberbkg, bulkbkg_mode = None):
    """
    mode: None, direct, constant, fit_q
    direct means using the direct wide angle line for substraction
    constant means using the constant value for substraction
    fit means fitting the wide angle over Q to be used as a wide angle bkg, requires alpha and wavelength in metadata of the sample
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
    if check_keys_numeric(["flux", "cttime"], sampledata["metadata"], chamberbkg["metadata"]):
        I0_sample_bkg = sampledata["metadata"]["flux"]*sampledata["metadata"]["cttime"] / (chamberbkg["metadata"]["flux"]*chamberbkg["metadata"]["cttime"])
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
        bulkbkg["intensity"] = bulkbkg_mode
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
                correcteddata["Intensity"] = data_chamber_subtracted[:,:bulk_qxy0_idx[0]] - np.outer(bulkbkg["Intensity"],np.ones((1,bulk_qxy0_idx[0])))
                correcteddata["error"] = np.sqrt(err_propogate[:,:bulk_qxy0_idx[0]]**2 + (np.outer(bulkbkg["error"],np.ones((1,bulk_qxy0_idx[0]))))**2)
                
                correcteddata["bulkbkg"] = bulkbkg
            else:
                # subtract the fit
                if "Q" in correcteddata:
                    bulkbkg_Q = np.mean(np.atleast_2d(correcteddata["Q"][:, bulk_qxy0_idx]), axis = 1) # this is the q axis
                    # fit Q
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


#def GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW):
def eCWM_analysis(GIXOS, transmission_corr = False, footprint_effect = False):
    """
    name changed to eCWM analysis
    use metadata imbedded in the input data, no DSbetaHW
    input for calc_film_DS_RRF_integ: use DSphi_HW instead of DSqxy_HW
    """
    GIXOS["fresnel"] = GIXOS_fresnel(GIXOS ["Qz"], GIXOS["metadata"]["Qc"]) # check if fresnel == GIXOS_fresnel      SAME
    #GIXOS["Qz_array"] = np.asarray(GIXOS ["Qz"]).reshape(-1, 1) # done to convert GIXOS ["Qz"] from a row vetor to a column vector for GIXOS_Tsqr
    # Qz should always be a column vector!
    if footprint_effect and ("footprint" in GIXOS["metadata"]) and ("Ddet" in GIXOS["metadata"]) and ("alpha_i" in GIXOS["metadata"]) and ("energy" in GIXOS["metadata"]):
        GIXOS["dQz"] = GIXOS_dQz(GIXOS["Qz"], GIXOS["metadata"]["energy"], GIXOS["metadata"]["alpha_i"], GIXOS["metadata"]["Ddet"], GIXOS["metadata"]["footprint"])     # Almost same, just not iterating through enough times(?) --> missing last row      SAME now
    else:
        print("No footprint broadending calculation. For calculation: please set footprint_effect = True and provide the footprint [mm], detector distance Ddet [mm], incident angle alpha_i [deg] and energy [eV] in metadata field")

    if transmission_corr and ("Ddet" in GIXOS["metadata"]) and ("alpha_i" in GIXOS["metadata"]) and ("energy" in GIXOS["metadata"]):
        if footprint_effect and ("footprint" in GIXOS["metadata"]):
            GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz"], GIXOS["metadata"]["Qc"], GIXOS["metadata"]["energy"], GIXOS["metadata"]["alpha_i"], GIXOS["metadata"]["Ddet"], GIXOS["metadata"]["footprint"])  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
        else:
            GIXOS["transmission"] = GIXOS_Tsqr(GIXOS["Qz"], GIXOS["metadata"]["Qc"], GIXOS["metadata"]["energy"], GIXOS["metadata"]["alpha_i"], GIXOS["metadata"]["Ddet"], 0.1)  #  Mostly the same, but the 4th column starts to deviate from the MATLAB output by hundredths
    else:
        GIXOS["transmission"] = np.ones((len(GIXOS["Qz"]),3))
        print("no transmission correction. For correction: please set the transmission_corr = True and provide the detector distance Ddet [mm], incident angle alpha_i [deg] and energy [eV] in metadata field")

    GIXOS["DS_RRF_integ"], GIXOS["DS_term_integ"], GIXOS["RRF_term_integ"] = calc_film_DS_RRF_integ(GIXOS["tt"], metadata["qxy0"][ metadata["qxy0_select_idx"] ], GIXOS["metadata"]["energy"]/1000, GIXOS["metadata"]["alpha_i"], GIXOS["metadata"]["RqxyHW"], metadata["DSqxyHW_real"], DSbetaHW, GIXOS["metadata"]["tension"], GIXOS["metadata"]["temperature"], GIXOS["metadata"]["kappa"], GIXOS["metadata"]["amin"], use_approx=False)
    # DS = Diffuse Scatter; RRF = Specular Reflectivity Normalized by Fresnel Reflectivity
    # Approx form is derived from Taylor expansion, which is dependent on being close to 0 angle --> higher deviations at high angles

    # computes reflectivity
    GIXOS["refl"] = np.column_stack([
        GIXOS["Qz"],
        GIXOS["GIXOS"] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["transmission"][:, 3],
        GIXOS["error"] / GIXOS["DS_RRF_integ"] * GIXOS["fresnel"][:, 1] / GIXOS["transmission"][:, 3],
        GIXOS["dQz"][:, 4]
    ])

    # computes structure factor 
    GIXOS["SF"] = np.column_stack([
        GIXOS["Qz"],
        GIXOS["GIXOS"] / GIXOS["DS_term_integ"] / GIXOS["transmission"][:, 3],
        GIXOS["error"] / GIXOS["DS_term_integ"] / GIXOS["transmission"][:, 3],
        GIXOS["dQz"][:, 4]
    ])
    return GIXOS # outputs GIXOS with reflectivity and structure factor added as new columns


def rect_slit_function(GIXOS, metadata):
    xrr_data = pd.read_csv(metadata["path_xrr"] + metadata["xrr_datafile"], delim_whitespace=True)
    xrr_config = {
        'energy' : 14400,
        'sdd' : 1039.9,
        'slit_h' : 1,
        'slit_v' : 0.66, 
    }

    xrr_config["wavelength"] = 12400/xrr_config["energy"]
    xrr_config["wave_number"] = 2 * np.pi / xrr_config["wavelength"]
    xrr_config["Qz"] = GIXOS["Qz"]
    xrr_config["dataQz"] = xrr_data.iloc[:, 0].astype(float).to_numpy()  # see later bc accessing data !!!!!!!!!!!!!!!!!!!  might have bugs bc came out as strings and need to convert to float
    xrr_config["beta_xrr"] = np.degrees(np.arcsin(xrr_config["Qz"] / 2 / xrr_config["wave_number"]))
    xrr_config["beta_xrr"] = xrr_config["beta_xrr"].reshape(-1, 1) # do this, otherwise beta_xrr has shape (46,) instead of (46, 1) which will mess up xrr_config_phi_array_for_qxy_slit_min and make it (46, 46) instead of (46, 1) like MATLAB code
    xrr_config["dataRF"] = (( xrr_config["dataQz"] - np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2)) / (xrr_config["dataQz"] + np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2))) * np.conj((xrr_config["dataQz"] - np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2)) / (xrr_config["dataQz"] + np.lib.scimath.sqrt(xrr_config["dataQz"]**2 - metadata["Qc"]**2)))   # needed for final file conversion
    xrr_config["RF"] = GIXOS["fresnel"][:, 1]
    xrr_config["kbT_gamma"] = metadata["kb"] * metadata["temperature"] / metadata["tension"] * 10 ** 20
    xrr_config["eta"] = xrr_config["kbT_gamma"] / 2 / np.pi * xrr_config["Qz"] ** 2
    # maybe delete xrr_config for simplicity
    xrr_config["delta_phi_HW"] = np.degrees(np.arctan(xrr_config["slit_h"] / 2 / xrr_config["sdd"] / np.cos(np.radians(xrr_config["beta_xrr"]))))
    xrr_config["delta_beta_HW"] =   np.degrees(np.arcsin(xrr_config["slit_v"] / 2 / xrr_config["sdd"] * np.cos(np.radians(xrr_config["beta_xrr"]))))

    xrr_config["slit_h_coord"] = np.arange(-xrr_config["slit_h"] / 2, xrr_config["slit_h"] / 2 + 0.005, 0.005)
    xrr_config["slit_v_coord"] = np.arange(-xrr_config["slit_v"] / 2, xrr_config["slit_v"] / 2 + 0.005, 0.005)
    xrr_config["slit_t"] = np.column_stack((xrr_config["slit_h_coord"], np.ones(len(xrr_config["slit_h_coord"])) * xrr_config["slit_v"] / 2))  
    xrr_config["slit_b"] = np.column_stack((xrr_config["slit_h_coord"], np.ones(len(xrr_config["slit_h_coord"])) * -xrr_config["slit_v"] / 2))
    xrr_config["slit_l"] = np.column_stack((np.ones(len(xrr_config["slit_v_coord"])) * -xrr_config["slit_h"] / 2, xrr_config["slit_v_coord"]))
    xrr_config["slit_r"] = np.column_stack((np.ones(len(xrr_config["slit_v_coord"])) * xrr_config["slit_h"] / 2, xrr_config["slit_v_coord"]))
    xrr_config["slit_coord"] = np.concatenate((xrr_config["slit_t"], xrr_config["slit_r"], np.flipud(xrr_config["slit_b"]), np.flipud(xrr_config["slit_l"])), axis = 0)
    xrr_config["qxy_slit"] = np.zeros( (xrr_config["slit_coord"].shape[0], 2, xrr_config["beta_xrr"].shape[0]) )
    xrr_config["qxy_slit_min"] = np.zeros( (xrr_config["beta_xrr"].shape[0], 1) )
    xrr_config["ang"] = np.arange(0, 2 * np.pi, 0.01)
    #xrr_config_ang = xrr_config_ang.reshape(-1, 1)  # transposing to make it a column vector
    xrr_config["qxy_slit_min_coord"] = np.zeros( (xrr_config["ang"].shape[0], 2, xrr_config["qxy_slit_min"].shape[0]))
    for idx in range(len(xrr_config["beta_xrr"])):
        xrr_config["qxy_slit"][:, :, idx] = xrr_config["wave_number"] * np.column_stack([xrr_config["slit_coord"][:, 0] / xrr_config["sdd"], xrr_config["slit_coord"][:, 1] / xrr_config["sdd"] * np.sin(np.radians(xrr_config["beta_xrr"][idx]))])
        xrr_config["qxy_slit_min"][idx, 0] = np.min(np.sqrt(xrr_config["qxy_slit"][:, 0, idx]**2 + xrr_config["qxy_slit"][:, 1, idx]**2))
        xrr_config["qxy_slit_min_coord"][:, :, idx] = xrr_config["qxy_slit_min"][idx] * np.column_stack([np.cos(xrr_config["ang"]), np.sin(xrr_config["ang"])])  # might not need np.array

    xrr_config["phi_max_qxy_slit_min"] = np.degrees(np.arctan(xrr_config["qxy_slit_min"] / xrr_config["wave_number"] / np.cos(np.radians(xrr_config["beta_xrr"]))))

    xrr_config["phi_array_for_qxy_slit_min"] = xrr_config["phi_max_qxy_slit_min"] * np.array([0, 1/5, 2/5, 3/5, 4/5])
    xrr_config["delta_beta_array_for_qxy_slit_min"] = np.degrees(
        np.arcsin(
            (np.sqrt(
                np.maximum(xrr_config["qxy_slit_min"][:, 0:1]**2 - 
                        (np.tan(np.radians(xrr_config["phi_array_for_qxy_slit_min"])) * 
                            np.cos(np.radians(xrr_config["beta_xrr"])) * 
                            xrr_config["wave_number"]) ** 2, 0))
            / (xrr_config["wave_number"] * np.sin(np.radians(xrr_config["beta_xrr"]))))
            * np.cos(np.radians(xrr_config["beta_xrr"]))
        )
    )

    delta_beta_HW_1d = xrr_config["delta_beta_HW"][:, 0]
    for idx in range(xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]):
        repidx = xrr_config["delta_beta_array_for_qxy_slit_min"][:, idx] >= delta_beta_HW_1d
        xrr_config["delta_beta_array_for_qxy_slit_min"][repidx, idx] = delta_beta_HW_1d[repidx]  # replace values that are greater than delta_beta_HW with delta_beta_HW

    xrr_config["phi_array_for_qxy_slit_min"] = np.hstack([xrr_config["phi_array_for_qxy_slit_min"], xrr_config["phi_max_qxy_slit_min"]]) # might not work with np.column_stack bc size mismatch --> column_hstack

    xrr_config["bkgoff"] = 1
    xrr_config["bkg_phi"] = np.degrees(np.arctan(xrr_config["bkgoff"] / (xrr_config["sdd"] * np.cos(np.radians(xrr_config["beta_xrr"])))))   # off by ten thousandths place - supposed to get larger as index increases, but decreases instead?


    xrr_config["r_step"] = 0.001
    xrr_config["r"] = np.sqrt(np.maximum(np.arange(0.001, 8*round(metadata["Lk"]) + xrr_config["r_step"], xrr_config["r_step"]) ** 2 + metadata["amin"] ** 2, 0))
    xrr_config["C_integrand"] = np.zeros((len(xrr_config["Qz"]), len(xrr_config["r"])))
    for idx in range(len(xrr_config["Qz"])):
        xrr_config["C_integrand"][idx, :] = 2 * np.pi * xrr_config["r"]**(1 - xrr_config["eta"][idx]) * (np.exp(-xrr_config["eta"][idx] * besselk(0, xrr_config["r"] / metadata["Lk"])) - 1)    # off by thousandths place
    # Matches up till here

    xrr_config["C"] = np.sum(xrr_config["C_integrand"], axis = 1) * xrr_config["r_step"]
    xrr_config["qxy_slit_min_flat"] = xrr_config["qxy_slit_min"].flatten()  # (46,) so that RRF_term does not return a (46, 46) array since MATLAB returns a (46, 1)
    xrr_config["RRF_term"] = (xrr_config["qxy_slit_min_flat"] ** xrr_config["eta"] + xrr_config["qxy_slit_min_flat"] ** 2 * xrr_config["C"] / 4 / np.pi) * (1/metadata["qmax"]) ** xrr_config["eta"] * np.exp(xrr_config["eta"] * besselk(0, 1 / metadata["Lk"] / metadata["qmax"]))
    xrr_config["specular_qxy_min"] = xrr_config["RF"] * xrr_config["RRF_term"] # off by hundredths/thousandths

    xrr_config["region_around_radial_u_r"] = np.zeros((len(xrr_config["beta_xrr"]), xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]))    # off by ~5-8 thousandths
    xrr_config["region_around_radial_l_r"] = np.zeros((len(xrr_config["beta_xrr"]), xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]))    # off by ~5-8 thousandths
    xrr_config["diff_r"] = np.zeros((len(xrr_config["beta_xrr"]), 1))    # VERY OFF
    xrr_config["diff_r_bkgoff"] = np.zeros((len(xrr_config["beta_xrr"]), 1))    # VERY OFF

    xrr_config["diff_r"] = xrr_config["diff_r"].flatten()
    xrr_config["diff_r_bkgoff"] = xrr_config["diff_r_bkgoff"].flatten()


    angle_factor = (np.pi / 180) ** 2 * (9.42e-6) ** 2 # taking it outside of the loop decreases computation time
    
    def process_idx(idx):
        beta = xrr_config["beta_xrr"][idx]
        sin_beta = np.sin(np.radians(beta))
        Lk_idx = metadata["Lk"] if np.isscalar(metadata["Lk"]) else metadata["Lk"][idx]  # Handles array or scalar Lk

        fun_film = lambda tt, tth: film_integral_delta_beta_delta_phi(
            tt, tth, xrr_config["kbT_gamma"], xrr_config["wave_number"],
            beta, Lk_idx, metadata["amin"]
        )

        upper_vals = []
        lower_vals = []

        for phi_idx in range(xrr_config["delta_beta_array_for_qxy_slit_min"].shape[1]):
            # Upper
            upper, _ = dblquad(
                lambda tth, tt: fun_film(tt, tth),
                xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_array_for_qxy_slit_min"][idx, phi_idx],
                xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx + 1],
                epsabs=1e-12, epsrel=1e-10
            )
            upper_vals.append(upper * angle_factor / sin_beta)

            # Lower
            lower, _ = dblquad(
                lambda tth, tt: fun_film(tt, tth),
                xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
                xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_array_for_qxy_slit_min"][idx, phi_idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx],
                lambda _: xrr_config["phi_array_for_qxy_slit_min"][idx, phi_idx + 1],
                epsabs=1e-12, epsrel=1e-10
            )
            lower_vals.append(lower * angle_factor / sin_beta)

        # diff_r
        result, _ = dblquad(
            func=fun_film,
            a=xrr_config["phi_max_qxy_slit_min"][idx],
            b=xrr_config["delta_phi_HW"][idx],
            gfun=lambda _: xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
            hfun=lambda _: xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
            epsabs=1e-8, epsrel=1e-6
        )
        diff_r = result * angle_factor / sin_beta

        # diff_r_bkgoff
        result2, _ = dblquad(
            func=fun_film,
            a=xrr_config["bkg_phi"][idx] - xrr_config["delta_phi_HW"][idx],
            b=xrr_config["bkg_phi"][idx] + xrr_config["delta_phi_HW"][idx],
            gfun=lambda _: xrr_config["beta_xrr"][idx] - xrr_config["delta_beta_HW"][idx],
            hfun=lambda _: xrr_config["beta_xrr"][idx] + xrr_config["delta_beta_HW"][idx],
            epsabs=1e-8, epsrel=1e-6
        )
        diff_r_bkgoff = result2 * angle_factor / sin_beta

        upper_vals = np.array(upper_vals, dtype=np.float64).flatten()
        lower_vals = np.array(lower_vals, dtype=np.float64).flatten()

        return idx, upper_vals, lower_vals, diff_r, diff_r_bkgoff

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_idx)(i) for i in range(len(xrr_config["beta_xrr"]))
    )

    # Fill in results
    for idx, upper_vals, lower_vals, diff_r, diff_r_bkgoff in results:
        xrr_config["region_around_radial_u_r"][idx, :] = upper_vals
        xrr_config["region_around_radial_l_r"][idx, :] = lower_vals
        xrr_config["diff_r"][idx] = diff_r
        xrr_config["diff_r_bkgoff"][idx] = diff_r_bkgoff
    
    xrr_config["Rterm_rect_slit"] = xrr_config["specular_qxy_min"] + 2*(np.sum(xrr_config["region_around_radial_u_r"] + xrr_config["region_around_radial_u_r"], axis = 1) + xrr_config["diff_r"])
    xrr_config["bkgterm_rect_slit"] = xrr_config["diff_r_bkgoff"]
    return xrr_config



def conversion_to_reflectivity(GIXOS, xrr_config):
    numerator_scaling = (xrr_config["Rterm_rect_slit"] - xrr_config["bkgterm_rect_slit"])
    denominator = GIXOS["DS_term_integ"] * GIXOS["transmission"][:, 3]  # Element-wise multiplication

    GIXOS["refl_recSlit"] = np.column_stack([
        GIXOS["Qz"],
        GIXOS["GIXOS"] * numerator_scaling / denominator,    # off by a bit
        GIXOS["error"] * numerator_scaling / denominator,    # off by a bit - see how to fix these 2
        GIXOS["dQz"][:, 4]                                               # issues are caused by numerator_scaling for sure (and it is both xrr_config_Rterm_rect_slit and xrr_config_bkgterm_rect_slit) 
    ])

    GIXOS["refl_roughness_term"] = (xrr_config["Rterm_rect_slit"] - xrr_config["bkgterm_rect_slit"]) / xrr_config["RF"]
    GIXOS["refl_roughness"] = np.sqrt(-np.log(GIXOS["refl_roughness_term"]) / GIXOS["Qz"]**2)
    GIXOS["SF"] = np.column_stack([GIXOS["SF"], GIXOS["refl_roughness"], GIXOS["refl_roughness_term"]])
    return GIXOS


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



from pseudo_xrr.plots import GIXOS_data_plot, R_data_plot, R_pseudo_data_plot
def rectangular_slit(metadata_file = './testing_data/gixos_metadata.yaml'):     # can make this a main function to run the whole code and have a parameter be the text file
    importGIXOSdata, importbkg = load_data(metadata_file)
    metadata = load_metadata(metadata_file)
    importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
    importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)
    metadata = real_space_2theta(metadata)
    GIXOS, DSbetaHW = GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
    GIXOS_data_plot(GIXOS, metadata)
    GIXOS = GIXOS_RF_and_SF(GIXOS, metadata, DSbetaHW)
    xrr_config = rect_slit_function(GIXOS, metadata)
    GIXOS = conversion_to_reflectivity(GIXOS, xrr_config)
    print("xrr_config keys:", xrr_config.keys())

    GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step)
    R_data_plot(GIXOS, metadata, xrr_config)
    R_pseudo_data_plot(GIXOS, metadata, xrr_config)





import copy

def create_dependency_models(GIXOS, metadata, DSbetaHW):
    model = {
        "tt": np.ones(len(metadata['qz_selected'])),
        "Qz": np.ones(len(metadata['qz_selected'])),
        "Qxy": np.zeros((len(metadata['qz_selected']), GIXOS["Qxy"].shape[1]))
    }

    for idx in range(len(metadata['qz_selected'])):
        rowidx_arr = np.where(GIXOS["Qz"] <= metadata['qz_selected'][idx])[0]
        rowidx = rowidx_arr[-1]
        model["tt"][idx] = GIXOS["tt"][rowidx]
        model["Qz"][idx] = GIXOS["Qz"][rowidx]
        model["Qxy"][idx, :] = GIXOS["Qxy"][rowidx, :]

    assume_model = {
        "1": copy.deepcopy(model),
        "2": copy.deepcopy(model)
    }

    CWM_model = copy.deepcopy(model)

    model["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    model["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    model["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["1"]["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["1"]["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["1"]["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["2"]["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["2"]["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    assume_model["2"]["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    CWM_model["DS_RRF"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    CWM_model["DS_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))
    CWM_model["RRF_term"] = np.zeros((len(model["tt"]), GIXOS["GIXOS"].shape[1]))

    metadata["energy"] = np.asarray(metadata["energy"])
    metadata["alpha_i"] = np.asarray(metadata["alpha_i"])
    metadata["RqxyHW"] = np.asarray(metadata["RqxyHW"])
    metadata["DSqxyHW_real"] = np.asarray(metadata["DSqxyHW_real"])
    # GIXOS["DSbetaHW"] = np.asarray(GIXOS["DSbetaHW"])        only add this and make changes if we say that DSbetaHW is part of GIXOS above
    DSbetaHW = np.asarray(DSbetaHW)
    metadata["tension"] = np.asarray(metadata["tension"])
    metadata["temperature"] = np.asarray(metadata["temperature"])
    metadata["kappa"] = np.asarray(metadata["kappa"])
    metadata["amin"] = np.asarray(metadata["amin"])


    def process(idx):
        show_last_plot = (idx == GIXOS["GIXOS"].shape[1] - 1)  # Only show plot on last iteration
        model_DS_RRF, model_DS_term, model_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["qxy0"][idx], metadata['energy']/1000, metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], metadata['kappa'], metadata['amin'], show_plot = False)
        assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["qxy0"][idx], metadata['energy']/1000, metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], metadata['assume_kappa'][0], metadata['amin'], show_plot = False)
        assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["qxy0"][idx], metadata['energy']/1000, metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], metadata['assume_kappa'][1], metadata['amin'], show_plot = False)
        CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term = calc_film_DS_RRF_integ(model["tt"], metadata["qxy0"][idx], metadata['energy']/1000, metadata['alpha_i'], metadata['RqxyHW'], metadata['DSqxyHW_real'][idx], DSbetaHW, metadata['tension'], metadata['temperature'], 0, metadata['amin'], show_plot = show_last_plot)
        return idx, model_DS_RRF, model_DS_term, model_RRF_term, assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term, assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term, CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process)(i) for i in range(GIXOS["GIXOS"].shape[1])
    )

    for idx, model_DS_RRF, model_DS_term, model_RRF_term, assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term, assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term, CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term in results:
        model["DS_RRF"][:, idx], model["DS_term"][:, idx], model["RRF_term"][:, idx] = model_DS_RRF, model_DS_term, model_RRF_term
        assume_model["1"]["DS_RRF"][:, idx], assume_model["1"]["DS_term"][:, idx], assume_model["1"]["RRF_term"][:, idx] = assume_model_1_DS_RRF, assume_model_1_DS_term, assume_model_1_RRF_term
        assume_model["2"]["DS_RRF"][:, idx], assume_model["2"]["DS_term"][:, idx], assume_model["2"]["RRF_term"][:, idx] = assume_model_2_DS_RRF, assume_model_2_DS_term, assume_model_2_RRF_term
        CWM_model["DS_RRF"][:, idx], CWM_model["DS_term"][:, idx], CWM_model["RRF_term"][:, idx] = CWM_model_DS_RRF, CWM_model_DS_term, CWM_model_RRF_term
    return model, assume_model, CWM_model