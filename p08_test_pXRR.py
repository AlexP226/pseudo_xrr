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
from pseudo_xrr.data_io import *
from pyinstrument import Profiler

#%%
SF_file = "U:/p08/2023/data/11016139/shared/analysis_version1/pseudoXRR/pseudoXRR2/large2thetaBkg/pp4_edta_a_1_00137_SF.dat"
R_file = "U:/p08/2023/data/11016139/shared/analysis_version1/pseudoXRR/pseudoXRR2/large2thetaBkg/pp4_edta_a_1_00137_R.dat"

SF_ref = np.loadtxt(SF_file, skiprows=29)
R_ref = np.loadtxt(R_file, skiprows=28)

#%% directly load data from meta and GIXOS will be automatically extracted:
# alternatively, load_data, geometrical correction, extract_1dGIXOS, and provide metadata into this field
GIXOSdata, GIXOSbkg = load_gixos_from_meta('./testing_data/gixos-process_config_p08test.yaml') 

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
fig_GIXOS, ax_GIXOS = GIXOS_raw_plot(
    GIXOSdata_q,
    GIXOSbkg_q,
    GIXOS_ana,
    metadata=GIXOS_ana["metadata"],
    show=False
)

outfigname = make_filename(GIXOS_ana["metadata"], suffix="GIXOS.png")
fig_GIXOS.savefig(outfigname, dpi=300, bbox_inches="tight")

#%% export corrected GIXOS
outgixosfile = make_filename(GIXOS_ana["metadata"], suffix="gixos.h5")
export_gixos_nxs(GIXOS_ana, outgixosfile)

#%% load back GIXOS
GIXOS_back = load_gixos_nxs(outgixosfile)

#%% from here on the operation will directly add results into the original dictionary variable (shared memory)
#%% qxy dependence
_, qxy_dependence_fit = GIXOS_qxy_dependence(GIXOS_ana, GIXOS_back['metadata']['dependency']['qz_selected'], fit_kappa = True)

#%% processing pseudo
_ = GIXOS2R(GIXOS_ana, transmission_corr = True, footprint_effect=True, use_approx=True)

#%% ---- export configuration ----
configfilename = make_filename(GIXOS_ana["metadata"], suffix="cfg.yaml")
save_metadata_yaml(GIXOS_ana["metadata"], configfilename)

#%% ------export orso -----------------
jsonfilename = "U:/p08/2023/data/11016139/beamtime-metadata-11016139.json"
fiofilename = "U:/p08/2023/data/11016139/raw/pp4_edta_a_1_00137.fio"

Rfilename = make_filename(GIXOS_ana["metadata"], suffix="R.ort")
_, dataset1 = export_orso(
    GIXOS_ana,
    which="refl",
    exportpath=Rfilename,
    json_path=jsonfilename,
    fio_path=fiofilename,
)

SFfilename = make_filename(GIXOS_ana["metadata"], suffix="SF.ort")
_, dataset2 = export_orso(
    GIXOS_ana,
    which="SF",
    exportpath=SFfilename,
    json_path=jsonfilename,
    fio_path=fiofilename,
)

