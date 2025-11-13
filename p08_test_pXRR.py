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

from pseudo_xrr.gixos import *
from pseudo_xrr.data_io import *
#from pseudo_xrr.slit import Rectungular_slit
#from pseudo_xrr.Dependency import *

from pyinstrument import Profiler

#%%
metadata = load_metadata('./testing_data/p08test_gixos_metadata.yaml')
#%%
datafileprefix = metadata['sample']+'_{:05d}'.format(metadata['scan'])+'_angle'
GIXSdata = load_data(datafileprefix, metadata['path'], datatype=metadata['datatype'])
bkgfileprefix = metadata['bkgsample']+'_{:05d}'.format(metadata['bkgscan'])+'_angle'
GIXSbkg = load_data(bkgfileprefix, metadata['path'], datatype=metadata['datatype'])

#%% geometric correction since my GIXS rebinning did not have this correction
GIXSdata = geometrical_corr(GIXSdata, Ddet = 560.7, det_px = 0.075, HWtth = 0.004, HWtt = 0.004)
GIXSbkg = geometrical_corr(GIXSbkg, Ddet = 560.7, det_px = 0.075, HWtth = 0.004, HWtt = 0.004)
#%%
GIXOSdata = extract_1dGIXOS(GIXSdata, metadata['tth'], metadata['DSpxHW'])
GIXOSbkg = extract_1dGIXOS(GIXSbkg, metadata['tth'], metadata['DSpxHW'])

#%%
GIXOSdata= binning_GIXOS_tt(GIXOSdata)
GIXOSbkg= binning_GIXOS_tt(GIXOSbkg)

#%%
GIXOSdata= remove_negative_2theta(GIXOSdata)
GIXOSbkg= remove_negative_2theta(GIXOSbkg)

#%%
GIXOSdata['metadata'] = {
                            "instrument": {"energy": 15000, "alpha_i": 0.07},
                            "qxy0": metadata["qxy0"],
                            "qxy_bkg": 0.3,
                            }
GIXOSbkg['metadata'] = {
                            "instrument": {"energy": 15000, "alpha_i": 0.07},
                            "qxy0": metadata["qxy0"],
                            "qxy_bkg": 0.3,
                            }
GIXOSdata_q = GIXOS_th2q(GIXOSdata)
GIXOSbkg_q = GIXOS_th2q(GIXOSbkg)
#%%
GIXOS_clean = GIXOS_background_corr(GIXOSdata_q, GIXOSbkg_q, bulkbkg_mode = "direct")


#%%
importGIXOSdata, importbkg = load_data_from_meta('./testing_data/p08test_gixos_metadata.yaml')

#%%
importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
#%%
# Create the plot
plt.figure()
plt.plot(GIXOSdata["tt"], GIXOSdata["Intensity"][:,2])
plt.plot(GIXOSbkg["tt"], GIXOSbkg["Intensity"][:,2])
plt.plot(GIXOSdata["tt"], GIXOSdata["Intensity"][:,2]-GIXOSbkg["Intensity"][:,2])
plt.errorbar(GIXOS_clean["tt"], GIXOS_clean["Intensity"][:,2], yerr = GIXOS_clean["error"][:,2], fmt ='o', markersize = 1, capsize = 3)
plt.plot(GIXOS_clean["bulkbkg"]["tt"], GIXOS_clean["bulkbkg"]["Intensity"])
# Set log10 scale on the y-axis
#plt.yscale('log')
# Add labels and title
plt.xlabel("X values")
plt.ylabel("Y values (log scale)")
plt.ylim([0, 20000])
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()
#%%
dependency_wrapper('./testing_data/gixos_metadata.yaml')
#%%