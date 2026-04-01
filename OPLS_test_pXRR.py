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
GIXOSdata, GIXOSbkg = load_gixos_from_meta('./testing_data/gixos-process_config_OPLStest.yaml') 

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
fig_GIXOS, ax_GIXOS = GIXOS_raw_plot(
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

#%% ---- export configuration ----
configfilename = make_filename(GIXOS_ana["metadata"], suffix="cfg.yaml")
save_metadata_yaml(GIXOS_ana["metadata"], configfilename)


#%%
#%% ------export orso -----------------
Rfilename = make_filename(GIXOS_ana["metadata"], suffix="R.ort")
_, dataset1 = export_orso(
    GIXOS_ana,
    which="refl",
    exportpath=Rfilename,
)

SFfilename = make_filename(GIXOS_ana["metadata"], suffix="SF.ort")
_, dataset2 = export_orso(
    GIXOS_ana,
    which="SF",
    exportpath=SFfilename,
)
