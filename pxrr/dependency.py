from pxrr.data_io import load_data, load_metadata, binning_GIXOS_data, remove_negative_2theta, dependency_real_space_2theta, GIXOS_data_plot_prep, GIXOS_RF_and_SF, rect_slit_function, conversion_to_reflectivity, GIXOS_file_output, Qxy_GIXOS_data_plot_prep, create_dependency_models
from pxrr.plots import GIXOS_vs_Qz_plot, GIXOS_vs_Qxy_plot, dependency_plot
def dependency_wrapper(metadata_file = './testing_data/gixos_metadata.yaml'):
    importGIXOSdata, importbkg = load_data(metadata_file)
    metadata = load_metadata(metadata_file)
    importGIXOSdata, importbkg = binning_GIXOS_data(importGIXOSdata, importbkg)
    importGIXOSdata, importbkg, tt_step = remove_negative_2theta(importGIXOSdata, importbkg)
    metadata = dependency_real_space_2theta(metadata)
    GIXOS, DSbetaHW = Qxy_GIXOS_data_plot_prep(importGIXOSdata, importbkg, metadata, tt_step)
    GIXOS_vs_Qz_plot(GIXOS, metadata)
    GIXOS_vs_Qxy_plot(GIXOS, metadata)
    model, assume_model, CWM_model = create_dependency_models(GIXOS, metadata, DSbetaHW)
    dependency_plot(GIXOS, metadata, model, assume_model, CWM_model)
    return

# load all data - also for slit
# input check
# models
# dependency plot