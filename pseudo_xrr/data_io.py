from ruamel.yaml import YAML
import numpy as np
import pandas as pd
from scipy.constants import pi, Boltzmann as kb
import os
import platform
from p08_GIXD.p08_GIXD import *
from pseudo_xrr.helpers import *
from pseudo_xrr.GIXOS import *
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



#%% 
# -----------------------------------------------------------------------------
# input block
# -----------------------------------------------------------------------------

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
    meta['tth']= np.degrees(np.arcsin(meta["qxy0"] * meta["instrument"]["wavelength"] / 4 / pi)) * 2
    meta["sample_params"]["rho_b"] = meta["sample_params"]["Qc"]**2/16/pi
    # RFscaling exactly as in the original script
    meta['I0'] = meta['measurements']['flux'] * meta['measurements']['cttime_sample']
    
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
    
    if "2d gixs" in datatype.lower():
        # checking facility otherwise raise error
        if meta["facility"] == "PETRA III/P08":
            filepattern = f"{sample}_{scan:05d}_angle"
            bkgfilepattern = f"{bkgsample}_{bkgscan:05d}_angle"
            importGIXOSdata = load_data(filepattern, path, metadata = yaml_path, datatype=datatype)
            importbkg = load_data(bkgfilepattern, path, metadata = yaml_path, datatype=datatype)
        else:
            raise ValueError(
                "2d gixs mode has only been implemented for PETRA III/P08 'Langmuir GID setup'"
            )
        
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
        if meta["facility"] == "NSLS-II/12ID":
            # build one prefix per scan id:
            # <sample>-id<id>
            sample_prefix_list = [f"{sample}-id{int(scan_id)}" for scan_id in scan]
            bkg_prefix_list = [f"{bkgsample}-id{int(scan_id)}" for scan_id in bkgscan]

            print("load 1d gixos cuts from NSLS-II/12ID")

            importGIXOSdata = load_data(
                sample_prefix_list,
                path,
                metadata=yaml_path,
                datatype=datatype
            )

            importbkg = load_data(
                bkg_prefix_list,
                path,
                metadata=yaml_path,
                datatype=datatype
            )
        else:
            raise ValueError(
                "1d gixos mode has only been implemented for NSLS-II/12ID."
            )
    
    else:
        print("only 2d gixs or 1d gixos cut is supported")
    
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
        importeddata["HWtth"] = mean1d_if_within_percent(np.diff(importeddata["tth"], axis = 1)/2)
        importeddata["HWpx_h"] = .5
        importeddata["HWtt"] = mean1d_if_within_percent(np.diff(importeddata["tt"], axis = 0)/2)
        importeddata["HWpx_v"] = .5
    
    elif "1d gixos" in datatype.lower():
        """
        implemented for NSLS-II/12ID 1d GIXOS cuts

        expected input:
        - gixosdataprefix can be either:
            * a single string prefix, or
            * a list/tuple/ndarray of string prefixes
        - each file is: <path><prefix>.txt

        expected file format:
        four columns: idx, tt(beta), intensity, qz
        """
        # normalize input to a list of file prefixes
        if isinstance(gixosdataprefix, str):
            prefix_list = [gixosdataprefix]
        else:
            prefix_list = list(gixosdataprefix)

        if len(prefix_list) == 0:
            raise ValueError("For '1d gixos', gixosdataprefix must contain at least one file prefix.")

        # load metadata early because we need tth / HWtth
        meta_loaded = None
        if metadata is not None:
            try:
                meta_loaded = load_metadata(metadata)
            except Exception:
                meta_loaded = None
                print("cannot find metadata")

        ncols = len(prefix_list)
        
        # optional consistency checks against metadata
        if meta_loaded is not None:
            if "tth" not in meta_loaded:
                raise ValueError("metadata must contain 'tth' for '1d gixos' loading.")

            if len(meta_loaded["tth"]) != ncols:
                raise ValueError(
                    "Number of 1d gixos files must match len(metadata['tth']) / len(metadata['qxy0'])."
                )

            if "qxy0" in meta_loaded and len(meta_loaded["qxy0"]) != ncols:
                raise ValueError(
                    "Number of 1d gixos files must match number of qxy0 entries in metadata."
                )        
        
        importeddata = None

        for idx, prefix in enumerate(prefix_list):
            GIXOSfilename = f"{path}{prefix}.txt"
            dataread = np.loadtxt(GIXOSfilename, skiprows=16)

            # expected columns: idx, tt(beta), intensity, qz
            tt_col = np.asarray(dataread[:, 1], dtype=float)
            inten_col = np.asarray(dataread[:, 2], dtype=float)

            if importeddata is None:
                nrows = len(tt_col)

                importeddata = {
                    "Intensity": np.zeros((nrows, ncols), dtype=float),
                    "error": np.zeros((nrows, ncols), dtype=float),
                    "tt": tt_col.copy(),                 # final tt should be 1D: (n,)
                    "tth": np.zeros((1, ncols), dtype=float),
                    "HWtth": None,
                    "HWpx_h": 0.5,
                    "HWtt": None,
                    "HWpx_v": 0.5,
                }

                # HWtth from metadata['HWtth'] -> shape (1,1)
                if meta_loaded is not None and "tth" in meta_loaded:
                    importeddata["tth"] = meta_loaded["tth"][np.newaxis, :]
                else:
                    raise ValueError(
                        "For '1d gixos', metadata must provide 'qxy0', 'energy', so tth can be calculated and the output matches extract_1dGIXOS."
                    )

                # HWtt from averaged step size of the tt column -> shape (1,)
                importeddata["HWtt"] = np.array([np.mean(np.diff(tt_col)) / 2], dtype=float)

                # HWtth from metadata['HWtth'] -> shape (1,1)
                if meta_loaded is not None and "HWtth" in meta_loaded["instrument"]:
                    importeddata["HWtth"] = np.array([[float(meta_loaded["instrument"]["HWtth"])]], dtype=float)
                else:
                    raise ValueError(
                        "For '1d gixos', metadata must provide 'HWtth' so the output matches extract_1dGIXOS."
                    )

                # optional consistency check for later files
                tt_ref = tt_col.copy()
            else:
                if len(tt_col) != importeddata["Intensity"].shape[0]:
                    raise ValueError(
                        f"1d gixos files do not have the same number of rows: "
                        f"{prefix} has {len(tt_col)}, expected {importeddata['Intensity'].shape[0]}"
                    )

                # require same tt grid
                if not np.allclose(tt_col, tt_ref, rtol=0, atol=1e-8):
                    raise ValueError(
                        f"1d gixos files do not share the same tt axis: {prefix}"
                    )

            importeddata["Intensity"][:, idx] = inten_col
            importeddata["error"][:, idx] = np.sqrt(np.maximum(inten_col, 0.0))

            # tth comes from metadata qxy0 list, shape must be (1, m)
            if metadata is not None:
                try:
                    if "meta_loaded" in locals() and meta_loaded is not None:
                        importeddata["metadata"] = meta_loaded
                    else:
                        importeddata["metadata"] = load_metadata(metadata)
                except Exception:
                    importeddata["metadata"] = None
                    print("cannot find metadata")
            else:
                raise ValueError("metadata is required for '1d gixos' loading.")


    else:
        print("only 2d gixs (PETRA III/P08) or 1d gixos cut (NSLS-II/12ID) is supported")
    
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

#%%
# ----------------------------------------------------------------------------
#   output block
# ----------------------------------------------------------------------------


def GIXOS_file_output(GIXOS, xrr_config, metadata, tt_step):
    xrrfilename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_R_PYTHON_TEST.dat" # becomes "instrument_46392_R_PYTHON.dat" - qz and dqz columns are very accurate, but R and dR start off semi-accurate but increasingly deviate after ~15th value
    with open(xrrfilename, 'w') as f:
        f.write(f"# files\n")
        f.write(f"sample file: {metadata['sample']}-id{metadata['scan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"background file: {metadata['bkgsample']}-id{metadata['bkgscan'][ metadata['qxy0_select_idx'] ]}\n")
        f.write(f"wide angle bkg at qxy0 = {metadata['qxy_bkg']:.6f} /A\n")
        f.write(f"# geometry\n")
        f.write(f"energy [eV]: {metadata['energy']:.2f}\n")
        f.write(f"incidence [deg]: {metadata['alpha']}\n")
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
        f.write(f"incidence [deg]: {metadata['alpha']}\n")
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
        f.write(f"incidence [deg]: {metadata['alpha']}\n")
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
