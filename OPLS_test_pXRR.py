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

#%% routine 2: directly load data from meta and GIXOS will be automatically extracted:
GIXOSdata, GIXOSbkg = load_gixos_from_meta('./testing_data/OPLStest_gixos_metadata.yaml') 

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
#%% background subtraction
GIXOS_ana = GIXOS_background_corr(GIXOSdata_q, GIXOSbkg_q, bulkbkg_mode = 0, bulkbkg_const_mode= 1, bulkbkg_const_qz_lb= 0.7)

#%%
fig_GIXOS, ax_GIXOS = GIXOS_background_corr_plot(
    GIXOSdata_q,
    GIXOSbkg_q,
    GIXOS_ana,
    metadata=GIXOS_ana["metadata"],
    show=False
)

outfile = make_filename(GIXOS_ana["metadata"], suffix="GIXOS.png")
fig_GIXOS.savefig(outfile, dpi=300, bbox_inches="tight")


#%% from here on the operation will directly add results into the original dictionary variable (shared memory)
#%% qxy dependence
_, qxy_dependence_fit = GIXOS_qxy_dependence(GIXOS_ana, GIXOS_ana['metadata']['dependency']['qz_selected'], row_window=3, fit_kappa = True)

#%% processing pseudo
_ = GIXOS2R(GIXOS_ana, transmission_corr = True, footprint_effect=False, use_approx=True)
# #%%
# RFscaling_ref = GIXOS_ana["metadata"]['I0'] * GIXOS_ana["metadata"]['sample_params']['rho_b'] ** 2 / np.sin(np.radians(GIXOS_ana["metadata"]['instrument']['alpha'])) *GIXOS_ana['talpha_sqr']

# # Create the plot
# plt.figure()
# plt.errorbar(GIXOS_ana["refl"][:,0], GIXOS_ana["refl"][:,1]/GIXOS_ana["fresnel"][:,1], GIXOS_ana["refl"][:,2]/GIXOS_ana["fresnel"][:,1], fmt ='o', markersize = 1, capsize = 3, label = 'pseudoR')
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
