# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:39:16 2025

@author: shenc
"""
# NEED TO HAVE DATA FILES DOWNLOADED AND UPDATE PATHS
import math
import numpy as np
from numpy import trapz
import pandas as pd
import matplotlib.pyplot as plt

#from p08_GIXD import *
#from p08_general import *

from pseudo_xrr.eCWM import *
from pseudo_xrr.data_io import *
#from pseudo_xrr.slit import Rectungular_slit
#from pseudo_xrr.Dependency import *

from pyinstrument import Profiler

#%%
SF_file = "U:/p08/2023/data/11016139/shared/analysis_version1/pseudoXRR/pseudoXRR2/large2thetaBkg/pp4_edta_a_1_00137_SF.dat"
R_file = "U:/p08/2023/data/11016139/shared/analysis_version1/pseudoXRR/pseudoXRR2/large2thetaBkg/pp4_edta_a_1_00137_R.dat"

SF_ref = np.loadtxt(SF_file, skiprows=29)
R_ref = np.loadtxt(R_file, skiprows=28)

#%% routine 1: load metadata and data separately, and extract the GIXOS
metadata = load_metadata('./testing_data/p08test_gixos_metadata.yaml')
datafileprefix = metadata['measurements']['sample']+'_{:05d}'.format(metadata['measurements']['scan'])+'_angle'
GIXSdata = load_data(datafileprefix, metadata['paths']['gixs_path'], datatype=metadata['datatype'])
bkgfileprefix = metadata['measurements']['bkgsample']+'_{:05d}'.format(metadata['measurements']['bkgscan'])+'_angle'
GIXSbkg = load_data(bkgfileprefix, metadata['paths']['gixs_path'],  datatype=metadata['datatype'])
#%% geometric correction since my GIXS rebinning did not have this correction
GIXSdata = geometrical_corr(GIXSdata, Ddet = metadata["instrument"]["Ddet"], det_px = metadata["instrument"]["pixel"], HWtth =  GIXSdata["HWtth"][0,0], HWtt =  GIXSdata["HWtt"][0])
GIXSbkg = geometrical_corr(GIXSbkg, Ddet = metadata["instrument"]["Ddet"], det_px = metadata["instrument"]["pixel"], HWtth = GIXSbkg["HWtth"][0,0], HWtt =  GIXSbkg["HWtt"][0])
GIXOSdata = extract_1dGIXOS(GIXSdata, metadata['tth'], HWpx_h = metadata['DSpxHW'])
GIXOSbkg = extract_1dGIXOS(GIXSbkg, metadata['tth'], HWpx_h = metadata['DSpxHW'])
#% still need to populate metadata with instrument, sample parameters, and so on for th2q, bkg correction, eCWM analysis
GIXOSdata['metadata'] = metadata
GIXOSbkg['metadata'] = metadata
# #% if metadata is entered manually for further processing
# GIXOSdata['metadata'] = {
#                             "instrument": {"energy": 15000, "alpha_i": 0.07},
#                             "sample_params": {"Qc": 0.0218, "temperature": 295, "tension": 0.038, "kappa": 10, "amin": 5},
#                             "qxy0": metadata["qxy0"],
#                             "qxy_bkg": 0.3,
#                             "PseudoR": {"qxy0_select_idx": 1, 'RqxyHW': 0.0002},
#                             }
# GIXOSbkg['metadata'] = {
#                             "instrument": {"energy": 15000, "alpha_i": 0.07},
#                             "qxy0": metadata["qxy0"],
#                             "qxy_bkg": 0.3,
#                             }
#%% routine 2: directly load data from meta and GIXOS will be automatically extracted:
# GIXOSdata, GIXOSbkg = load_gixos_from_meta('./testing_data/p08test_gixos_metadata.yaml') 

#%% from here identical 
#%%binning in tt
GIXOSdata= binning_GIXOS_tt(GIXOSdata)
GIXOSbkg= binning_GIXOS_tt(GIXOSbkg)
#% remove negative 2theta
GIXOSdata= remove_negative_2theta(GIXOSdata)
GIXOSbkg= remove_negative_2theta(GIXOSbkg)
#% 2theta to q
GIXOSdata_q = GIXOS_th2q(GIXOSdata)
GIXOSbkg_q = GIXOS_th2q(GIXOSbkg)
#% background subtraction
GIXOS_clean = GIXOS_background_corr(GIXOSdata_q, GIXOSbkg_q, bulkbkg_mode = "fit", bulkbkg_offset_lb=0.9)

#%%
# Create the plot
plt.figure()
plt.plot(GIXOSdata["tt"], GIXOSdata["Intensity"][:,2])
plt.plot(GIXOSbkg["tt"], GIXOSbkg["Intensity"][:,2])
plt.plot(GIXOSdata["tt"], GIXOSdata["Intensity"][:,2]-GIXOSbkg["Intensity"][:,2])
plt.errorbar(GIXOS_clean["tt"], GIXOS_clean["Intensity"][:,2], yerr = GIXOS_clean["error"][:,2], fmt ='o', markersize = 1, capsize = 3)
plt.plot(GIXOS_clean["tt"], GIXOS_clean["bulkbkg"]["Intensity_at_GIXOS"][:,2])
# Set log10 scale on the y-axis
#plt.yscale('log')
# Add labels and title
plt.xlabel("X values")
plt.ylabel("Y values (log scale)")
plt.ylim([0, 20000])
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

#%% qxy dependence
qxy_dependence_predict = GIXOS_qxy_dependence(GIXOS_clean, GIXOS_clean['metadata']['dependency']['qz_selected'], fit_kappa = False)
qxy_dependence_fit = GIXOS_qxy_dependence(GIXOS_clean, GIXOS_clean['metadata']['dependency']['qz_selected'], fit_kappa = True)

#%% processing pseudo
GIXOS_ana = GIXOS2R(GIXOS_clean, transmission_corr = True, use_approx=False)

#%%
# Create the plot
plt.figure()
plt.plot(SF_ref[:,0],SF_ref[:,1]/GIXOS_ana["metadata"]["RFscaling"]/ (math.pi / 180) ** 2, label = 'matlab structure factor')
plt.errorbar(GIXOS_ana["refl"][:,0], GIXOS_ana["refl"][:,1]/GIXOS_ana["fresnel"][:,1]/GIXOS_ana["metadata"]["I0"], GIXOS_ana["refl"][:,2]/GIXOS_ana["fresnel"][:,1]/GIXOS_ana["metadata"]["I0"], fmt ='o', markersize = 1, capsize = 3, label = 'pseudoR')
plt.plot(R_ref[:,0],R_ref[:,1]/(0.0218/2/R_ref[:,0])**4/GIXOS_ana["metadata"]["RFscaling"]/ (math.pi / 180) ** 2, label = 'matlab pseudoR')
plt.errorbar(GIXOS_ana["SF"][:,0], GIXOS_ana["SF"][:,1]/GIXOS_ana["metadata"]["I0"], GIXOS_ana["SF"][:,2]/GIXOS_ana["metadata"]["I0"], fmt ='o', markersize = 1, capsize = 3, label = 'structure factor')
# Set log10 scale on the y-axis
plt.yscale('log')
# Add labels and title
plt.xlabel("X values")
plt.ylabel("Y values rad (log scale)")
plt.ylim([1e-7, 10])
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

#%% test rectangular slit
xrrqz = GIXOS_ana["refl"][:,0]
Psi_slit_approx = calc_eCWM_roughness_factor_SP(xrrqz, 
    energy=14400,
    sdd=1039.9,
    resolution_mode=1,
    resolution=[0.33,0.5],
    bkg_mode=0,
    bkg_off= 1,
    tension=0.038,
    temp=295,
    kappa=15,
    amin=5,
    use_approx=True)
sigma_slit_approx = np.sqrt(-1/xrrqz**2*np.log(Psi_slit_approx))

#%% work for qxy dependence
# np.savetxt("D:/intensity.dat", qxy_dependence["I_sum"])
# np.savetxt("D:/Qxy.dat", qxy_dependence["Qxy"])

