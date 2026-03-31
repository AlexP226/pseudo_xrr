# from data_io import load_metadata, etc., etc. maybe
import matplotlib.pyplot as plt
import os
import numpy as np

def GIXOS_data_plot(GIXOS, metadata):
    plt.figure(figsize = (10, 6))
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS_raw"] - GIXOS["GIXOS_bkg"], 'ko', markersize=3, label='corrected data by chamber transmission bkg')
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS"], 'go', markersize=3, label='with large Qz subtraction')
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS_raw"], 'r-', linewidth=1.5, label='raw data')
    plt.plot(GIXOS["Qz"], GIXOS["GIXOS_bkg"], 'b-', linewidth=1.5, label='bkg data at same tth')
    plt.plot(GIXOS["Qz"], GIXOS["raw_largetth"], 'r:', linewidth=1.5, label='raw data wide angle')
    plt.plot(GIXOS["Qz"], GIXOS["bkg_largetth"], 'b:', linewidth=1.5, label='bkg data wide angle')
    plt.axhline(0, color='k', linestyle='-.', linewidth=1.5, label='0-line')

    # --- Show legend and display
    plt.legend()
    plt.xlabel("Qz")
    plt.ylabel("Intensity")
    plt.title("GIXOS Data")
    plt.grid(True)
    plt.tight_layout()

    plt.xlim([0, 1.05])
    ax = plt.gca()

    ax.set_xlabel(r'$Q_z$ [$\mathrm{\AA}^{-1}$]', fontsize=12)  # Qz with angstrom symbol
    ax.set_ylabel('GIXOS', fontsize=12)

    ax.set_xticks(np.arange(0, 1.01, 0.2))  # same as 0:0.2:1
    ax.tick_params(axis='both', labelsize=12, width=2, direction='out')

    plt.legend(loc='upper right', frameon=False)  # 'NorthEast' => 'upper right'

    # --- Save figure
    filename = f"{metadata['sample']}_{metadata['qxy0_select_idx']:05d}_GIXOS.jpg"
    save_path = os.path.join(metadata["path_out"], filename)
    plt.savefig(save_path, dpi=300)
    plt.show()



def R_data_plot(GIXOS, metadata, xrr_config): # want to have xrr_config be a dict that the xrr_config function will return w/ vars we need
    GIXOS["refl_recSlit"] = np.array(GIXOS["refl_recSlit"])

    # Close any existing figure named 'refl'
    plt.close('refl')

    # Create new figure
    fig, ax = plt.subplots(num='refl', figsize=(6, 5))
    fig.canvas.manager.set_window_title('refl')

    # Plot error bars
    # Commented line corresponds to the MATLAB commented errorbar line
    # ax.errorbar(GIXOS['refl'][:,0], GIXOS['refl'][:,1]/RFscaling, yerr=GIXOS['refl'][:,2]/RFscaling,
    #             fmt='o', markersize=3,
    #             label=f"δQ_xy= {geo['RqxyHW']:.1e} Å⁻¹")

    ax.errorbar(
        GIXOS["refl_recSlit"][:, 0],
        GIXOS["refl_recSlit"][:, 1] / metadata["RFscaling"],
        yerr=GIXOS["refl_recSlit"][:, 2] / metadata["RFscaling"],
        fmt='o',
        markersize=5,
        label=(
            f"slit: {xrr_config['slit_v']:.2f} mm (v) x {xrr_config['slit_h']:.2f} mm (h); "
            f"{xrr_config['sdd']:.1f} mm, {xrr_config['energy'] / 1000:.1f} keV"
        )
    )

    ax.set_xlim(0, 0.8)
    ax.set_xlabel(r'$Q_z \; [\AA^{-1}]$', fontsize=14)
    ax.set_ylabel('R', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=14, direction='out')
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_yscale('log')
    ax.legend(loc='lower right', frameon=False)  # 'Southeast' → 'lower right' in matplotlib

    plt.tight_layout()

    # Save figure
    filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_R_PYTHON.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()






def R_pseudo_data_plot(GIXOS, metadata, xrr_config):
    GIXOS["refl_recSlit"] = np.array(GIXOS["refl_recSlit"])
    GIXOS["fresnel"] = np.array(GIXOS["fresnel"])
    xrr_data_data = np.array(metadata["xrr_data"])   # maybe add xrr_data to xrr_config dict

    # Close existing figure named 'RRF'
    plt.close('RRF')

    fig, ax = plt.subplots(num='RRF', figsize=(6, 5))
    fig.canvas.manager.set_window_title('RRF')

    # Plot first errorbar: refl_recSlit normalized by fresnel and RFscaling
    ax.errorbar(
        GIXOS["refl_recSlit"][:, 0],
        GIXOS["refl_recSlit"][:, 1] / GIXOS["fresnel"][:, 1] / metadata["RFscaling"],
        yerr=GIXOS["refl_recSlit"][:, 2] / GIXOS["fresnel"][:, 1] / metadata["RFscaling"],
        fmt='ro',
        markersize=5,
        linewidth=1.5,
        label=(
            f"slit: {xrr_config['slit_v']:.2f} mm (v) x {xrr_config['slit_h']:.2f} mm (h); "
            f"{xrr_config['sdd']:.1f} mm, {xrr_config['energy'] / 1000:.1f} keV"
        )
    )

    # Plot second errorbar: xrr_data normalized by dataRF
    ax.errorbar(
        xrr_data_data[:, 0],
        xrr_data_data[:, 1] / xrr_config["dataRF"],
        yerr=xrr_data_data[:, 2] / xrr_config["dataRF"],
        fmt='ko',
        markersize=5,
        linewidth=1.5,
        capsize=4,
        label='XRR 0.66mm slit'
    )

    ax.set_xlim(0, 0.8)
    ax.set_ylim(1e-5, 10)
    ax.set_xlabel(r'$Q_z \; [\AA^{-1}]$', fontsize=14)
    ax.set_ylabel('R/R_F [a.u.]', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=14, direction='out')
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_yscale('log')
    ax.legend(loc='lower left', frameon=False)  # MATLAB 'SouthWest' ≈ matplotlib 'lower left'

    plt.tight_layout()

    filename = f"{metadata['path_out']}{metadata['sample']}_{metadata['scan'][ metadata['qxy0_select_idx'] ]:05d}_RRF_PYTHON.jpg"
    plt.savefig(filename, dpi=300)
    plt.show()


def GIXOS_vs_Qz_plot(GIXOS, metadata):
    # Plot GIXOS vs Qz for each Qxy_0
    fig_refl = plt.figure("raw")
    for idx in range(GIXOS["GIXOS"].shape[1]):
        label = f"Q_xy,0 = {metadata['qxy0'][idx]:.2f}Å⁻¹"
        plt.plot(GIXOS["Qz"], GIXOS["GIXOS"][:, idx] - GIXOS["bkg"], '-', linewidth=1.5, label=label)
    plt.axhline(0, color='k', linestyle='-.', linewidth=1.5, label='0-line')
    plt.xlim([0, 1.05])
    plt.xlabel(r'$Q_z$ [$\mathrm{\AA}^{-1}$]', fontsize=12)
    plt.ylabel("R*", fontsize=12)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.legend(loc='upper right', frameon=False)
    ax = plt.gca()
    ax.tick_params(direction='out', width=2, labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{metadata['path_out']}{metadata['sample']}_{min(metadata['scan']):05d}_{max(metadata['scan']):05d}_raw_all.jpg")

def GIXOS_vs_Qxy_plot(GIXOS, metadata):
    # Plot GIXOS vs Qxy at selected Qz
    fig_refl_qxy = plt.figure("raw Qxy")
    for idx, qz_val in enumerate(metadata['qz_selected']):
        plotrowidx = np.where(GIXOS["Qz"] <= qz_val)[0]
        if len(plotrowidx) == 0:
            continue
        plotrowidx = plotrowidx[-1]
        start = max(plotrowidx - 3, 0)
        end = min(plotrowidx + 4, GIXOS["GIXOS"].shape[0])
        ydata = np.mean(GIXOS["GIXOS"][start:end, :] - GIXOS["bkg"], axis=0) * 100 ** idx
        label = f"Q_z = {GIXOS['Qz'][plotrowidx]:.2f}Å⁻¹"
        plt.plot(GIXOS["Qxy"][plotrowidx, :], ydata, 'o:', markersize=4, linewidth=1.5, label=label)
    plt.xlim([0.015, 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$Q_{xy}$ [$\mathrm{\AA}^{-1}$]', fontsize=12)
    plt.ylabel("GIXOS", fontsize=12)
    ax = plt.gca()
    ax.tick_params(direction='out', width=2, labelsize=12)
    plt.legend(loc='upper right', frameon=False, fontsize=12, reverse = True) # added reverse to match the order that the lines are plotted in the 2nd graph (also see last cell)
    plt.tight_layout()


def dependency_plot(GIXOS, metadata, model, assume_model, CWM_model):
    fig_refl = plt.figure(figsize=(9, 9))
    ax = fig_refl.add_subplot(111)
    legendtext = []

    # First loop: Plot experimental points
    for idx in range(len(model['Qz'])):
        plotrowidx = np.where(GIXOS['Qz'] <= metadata["qz_selected"][idx])[0][-1]
        
        intensity = np.mean(GIXOS['GIXOS'][plotrowidx-1:plotrowidx+2, :] - GIXOS['bkg'], axis=0)
        scale_factor = 4 / (180 / np.pi)**2 / metadata["RFscaling"] * metadata["rho_b"]**2 / np.sin(np.radians(metadata["alpha_i"])) * 100**idx

        ax.plot(
            GIXOS['Qxy'][plotrowidx, :],
            intensity * scale_factor,
            'o',
            markersize=7,
            markeredgecolor=metadata["colorset"][idx],
            linewidth=1.5
        )

        legendtext.append(f"Q_z = {GIXOS['Qz'][plotrowidx]:.2f} Å⁻¹")

    #colorset.reverse()  # Reverse the colorset to match the order of plots



    # Second loop: Plot model fits and comparisons
    for idx in range(len(model['Qz'])):
        plotrowidx = np.where(GIXOS['Qz'] <= metadata["qz_selected"][idx])[0][-1]

        mean_intensity = np.mean(GIXOS['GIXOS'][plotrowidx-1:plotrowidx+2, 0] - GIXOS['bkg'], axis=0)
        norm_factor = mean_intensity / metadata["RFscaling"] / model['DS_term'][idx, 0]
        scale_factor = 4 / (180 / np.pi)**2 * metadata["rho_b"]**2 / np.sin(np.radians(metadata["alpha_i"])) * 100**idx

        ax.plot(model['Qxy'][idx, :], model['DS_term'][idx, :] * norm_factor * scale_factor, 
                linewidth=2, color=metadata["colorset"][idx])

        for m_idx, assume in enumerate([assume_model["1"], assume_model["2"]]):
            term = assume['DS_term'][idx, :] * model['DS_term'][idx, -1] / assume['DS_term'][idx, -1]
            ax.plot(assume['Qxy'][idx, :], term * norm_factor * scale_factor,
                    linestyle='--', linewidth=2, color=metadata["colorset"][idx])

        cwm_term = CWM_model['DS_term'][idx, :] * model['DS_term'][idx, -1] / CWM_model['DS_term'][idx, -1]
        ax.plot(CWM_model['Qxy'][idx, :], cwm_term * norm_factor * scale_factor,
                'k-.', linewidth=0.5)

    # Formatting
    #legendtext.reverse()  # Reverse the legend text to match the order of plots

    ax.set_xlim([0.02, 0.6])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Qₓᵧ [Å⁻¹]', fontsize=14)
    ax.set_ylabel('I_DS', fontsize=14)
    ax.tick_params(labelsize=14, width=2, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(legendtext, loc='upper right', frameon=True, fontsize=14, reverse = True)
    ax.set_title(f"κ: {metadata['kappa']:.0f}±{metadata['kappa_deviation']:.0f} kbT, aₘᵢₙ: {metadata['amin']:.1f} Å")

    # Save the figure
    filename = f"{metadata['sample']}_{min(metadata['scan']):05d}_{max(metadata['scan']):05d}_Qxy.jpg"
    plt.savefig(os.path.join(metadata['path_out'], filename), dpi=300, bbox_inches='tight')