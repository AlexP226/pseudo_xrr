# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:39:16 2025

@author: shenc
"""
# NEED TO HAVE DATA FILES DOWNLOADED AND UPDATE PATHS
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from pseudo_xrr.eCWM import *
from pseudo_xrr.data_io import *
from pseudo_xrr.GIXOS import *
from pyinstrument import Profiler

#%%
SF_file = "U:/p08/2023/data/11016139/shared/analysis_version1/pseudoXRR/pseudoXRR2/large2thetaBkg/pp4_edta_a_1_00137_SF.dat"
R_file = "U:/p08/2023/data/11016139/shared/analysis_version1/pseudoXRR/pseudoXRR2/large2thetaBkg/pp4_edta_a_1_00137_R.dat"

SF_ref = np.loadtxt(SF_file, skiprows=29)
R_ref = np.loadtxt(R_file, skiprows=28)

#%% directly load data from meta and GIXOS will be automatically extracted:
# alternatively, load_data, geometrical correction, extract_1dGIXOS, and provide metadata into this field
GIXOSdata, GIXOSbkg = load_gixos_from_meta('./testing_data/p08test_gixos_metadata.yaml') 

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
GIXOS_ana = GIXOS_background_corr(GIXOSdata_q, GIXOSbkg_q, bulkbkg_mode = 2, bulkbkg_offset_lb=0.9)

#%%
# Create the plot
plt.figure()
plt.plot(GIXOSdata["tt"], GIXOSdata["Intensity"][:,2])
plt.plot(GIXOSbkg["tt"], GIXOSbkg["Intensity"][:,2])
plt.plot(GIXOSdata["tt"], GIXOSdata["Intensity"][:,2]-GIXOSbkg["Intensity"][:,2])
plt.errorbar(GIXOS_ana["tt"], GIXOS_ana["Intensity"][:,2], yerr = GIXOS_ana["error"][:,2], fmt ='o', markersize = 1, capsize = 3)
plt.plot(GIXOS_ana["tt"], GIXOS_ana["bulkbkg"]["Intensity_at_GIXOS"][:,2])
# Set log10 scale on the y-axis
#plt.yscale('log')
# Add labels and title
plt.xlabel("X values")
plt.ylabel("Y values (log scale)")
plt.ylim([0, np.max(GIXOSdata["Intensity"][20:,2])*2])
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

#%% from here on the operation will directly add results into the original dictionary variable (shared memory)
#%% qxy dependence
_, qxy_dependence_fit = GIXOS_qxy_dependence(GIXOS_ana, GIXOS_ana['metadata']['dependency']['qz_selected'], fit_kappa = True)

#%% processing pseudo
_ = GIXOS2R(GIXOS_ana, transmission_corr = True, footprint_effect=True, use_approx=True)

# #%%
# RFscaling_ref = GIXOS_ana["metadata"]['I0'] * GIXOS_ana["metadata"]['sample_params']['rho_b'] ** 2 / np.sin(np.radians(GIXOS_ana["metadata"]['instrument']['alpha'])) *GIXOS_ana['talpha_sqr']

# # Create the plot
# plt.figure()
# plt.plot(SF_ref[:,0],SF_ref[:,1]/RFscaling_ref/ (pi / 180) ** 2, label = 'matlab structure factor')
# plt.errorbar(GIXOS_ana["refl"][:,0], GIXOS_ana["refl"][:,1]/GIXOS_ana["fresnel"][:,1], GIXOS_ana["refl"][:,2]/GIXOS_ana["fresnel"][:,1], fmt ='o', markersize = 1, capsize = 3, label = 'pseudoR')
# plt.plot(R_ref[:,0],R_ref[:,1]/(0.0218/2/R_ref[:,0])**4/RFscaling_ref/ (pi / 180) ** 2, label = 'matlab pseudoR')
# plt.errorbar(GIXOS_ana["SF"][:,0], GIXOS_ana["SF"][:,1], GIXOS_ana["SF"][:,2], fmt ='o', markersize = 1, capsize = 3, label = 'structure factor')
# # Set log10 scale on the y-axis
# plt.yscale('log')
# # Add labels and title
# plt.xlabel("X values")
# plt.ylabel("Y values rad (log scale)")
# plt.ylim([1e-7, 10])
# plt.legend()
# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.show()


#%% work for qxy dependence
# np.savetxt("D:/intensity.dat", qxy_dependence["I_sum"])
# np.savetxt("D:/Qxy.dat", qxy_dependence["Qxy"])

