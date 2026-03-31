# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:46:59 2026

@author: shenc
"""
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# qxy dependence fit / predict and kappa value
# -----------------------------------------------------------------------------

def GIXOS_qxy_dependence_plot(
    results,
    *,
    metadata=None,
    show_refs=True,
    show_err=True,
    title=None
    ):
    """
    Plot extracted Qxy dependences together with CWM / eCWM reference curves.

    Parameters
    ----------
    results : dict
        Output dictionary from GIXOS_qxy_dependence().

    metadata : dict, optional
        Metadata dictionary. If provided and if
        metadata["sample_params"] contains the fields
        "tension", "temperature", and "amin", these are displayed
        on the plot.

    show_refs : bool, optional
        If True, plot the CWM / eCWM reference curves.

    show_err : bool, optional
        If True, plot the upper/lower eCWM error curves if available.

    title : str, optional
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(5, 7))

    for idx, qxy in enumerate(results['Qxy']):

        ax.plot(
            qxy,
            results['I_sum_offset'][idx, :],
            marker="o",
            linestyle="None",
            label=f"$Q_{{z}}≈{results['target_qz'][idx]:.2f}\\,\\AA^{{-1}}$"
        )

        if show_refs and results["ref_CWM"] is not None:
            ax.plot(
                qxy,
                results["ref_CWM"][idx, :],
                linestyle=":",
                color="k",
                linewidth=1.4
            )

        if show_refs and results["ref_eCWM"] is not None:
            ax.plot(
                qxy,
                results["ref_eCWM"][idx, :],
                linestyle="-",
                color="k",
                linewidth=1.8
            )

        if show_err and "ref_eCWM_err_u" in results and "ref_eCWM_err_l" in results:
            ax.plot(
                qxy,
                results["ref_eCWM_err_u"][idx, :],
                linestyle="--",
                color="k",
                linewidth=1.4
            )
            ax.plot(
                qxy,
                results["ref_eCWM_err_l"][idx, :],
                linestyle="--",
                color="k",
                linewidth=1.4
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_{xy}\,(\AA^{-1})$")
    ax.set_ylabel("(GI) diffuse intensity (offset)")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(r"$Q_{xy}$ dependence")

    # sample-parameter + scan annotation
    if metadata is not None:
        text_lines = []
        
        # --- scan id ---
        if "measurements" in metadata and metadata["measurements"] is not None:
            meas = metadata["measurements"]
            scan_val = meas.get("scan", None)

            if scan_val is not None:
                scan_arr = np.asarray(scan_val).ravel()

                if scan_arr.size == 1:
                    text_lines.append(f"scan Id = {int(scan_arr[0])}")
                elif scan_arr.size > 1:
                    text_lines.append(
                        f"scan Id = {int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
                    )
        # --- sample parameters ---
        if "sample_params" in metadata and metadata["sample_params"] is not None:
            sp = metadata["sample_params"]

            tension = sp.get("tension", None)
            temperature = sp.get("temperature", None)
            amin = sp.get("amin", None)

            if tension is not None:
                text_lines.append(rf"$\gamma = {float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
            if temperature is not None:
                text_lines.append(rf"$T = {float(temperature):.1f}\,\mathrm{{K}}$")
            if amin is not None:
                text_lines.append(rf"$a_{{\min}} = {float(amin):.1f}\,\AA$")

        if len(text_lines) > 0:
            ax.text(
                0.98, 0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none"
                )
            )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    fig.tight_layout()
    plt.show()

    return fig, ax


def GIXOS_RRF_plot(
    GIXOS,
    *,
    metadata=None,
    title=None,
    show=True
    ):
    """
    Plot pseudo-reflectivity / roughness-factor analysis from GIXOS2R output.

    This function visualizes the main outputs of `GIXOS2R()` on a semi-log plot
    (log10 intensity vs linear Qz), including:

    - normalized raw GIXOS intensity from the selected qxy0 column
    - structure factor |Phi(Qz)|^2
    - pseudo-reflectivity normalized by Fresnel reflectivity
    - specular roughness factor Psi_R(Qz)

    A vertical dashed line is also added at the Qz position where eta = 2.

    Parameters
    ----------
    GIXOS : dict
        Output dictionary from `GIXOS2R()`.

    metadata : dict, optional
        Metadata dictionary. If None, `GIXOS["metadata"]` is used.

    title : str, optional
        Plot title.

    show : bool, optional
        If True, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    ax : matplotlib.axes.Axes
        The axes object used for the plot.
    """
    if metadata is None:
        metadata = GIXOS.get("metadata", None)

    if metadata is None:
        raise ValueError("metadata is required either explicitly or in GIXOS['metadata'].")

    if "PseudoR" not in metadata or metadata["PseudoR"] is None:
        raise ValueError("metadata['PseudoR'] is required.")
    if "sample_params" not in metadata or metadata["sample_params"] is None:
        raise ValueError("metadata['sample_params'] is required.")

    pr = metadata["PseudoR"]
    samp = metadata["sample_params"]

    if "qxy0_select_idx" not in pr or pr["qxy0_select_idx"] is None:
        raise ValueError("metadata['PseudoR']['qxy0_select_idx'] is required.")

    qidx = int(pr["qxy0_select_idx"])
    Qc = float(samp["Qc"])

    # ------------------------------------------------------------
    # 1) structure factor |Phi|^2  (black scatter)
    # ------------------------------------------------------------
    qz_sf = np.asarray(GIXOS["SF"][:, 0], dtype=float)
    y_sf = np.asarray(GIXOS["SF"][:, 1], dtype=float)
    err_sf = np.asarray(GIXOS["SF"][:, 2], dtype=float)

    # ------------------------------------------------------------
    # 2) pseudo-reflectivity / Fresnel  (blue scatter)
    # ------------------------------------------------------------
    qz_refl = np.asarray(GIXOS["refl"][:, 0], dtype=float)
    fresnel = np.asarray(GIXOS["fresnel"][:, 1], dtype=float)

    y_refl_norm = np.asarray(GIXOS["refl"][:, 1], dtype=float) / fresnel
    err_refl_norm = np.asarray(GIXOS["refl"][:, 2], dtype=float) / fresnel

    # ------------------------------------------------------------
    # 3) normalized raw GIXOS intensity (red scatter)
    #    normalize at Qz closest to 2*Qc
    # ------------------------------------------------------------
    qz_raw = np.asarray(GIXOS["Qz"][:, qidx], dtype=float)
    y_raw = np.asarray(GIXOS["Intensity"][:, qidx], dtype=float)
    err_raw = np.asarray(GIXOS["error"][:, qidx], dtype=float)

    q_target = 2.0 * Qc
    idx_norm_raw = int(np.argmin(np.abs(qz_raw - q_target)))
    idx_norm_sf = int(np.argmin(np.abs(qz_sf - q_target)))

    if y_raw[idx_norm_raw] <= 0 or y_sf[idx_norm_sf] <= 0:
        raw_scale = 1.0
    else:
        raw_scale = y_sf[idx_norm_sf] / y_raw[idx_norm_raw]

    y_raw_norm = y_raw * raw_scale
    err_raw_norm = err_raw * raw_scale

    # ------------------------------------------------------------
    # 4) Psi_R (blue line)
    # ------------------------------------------------------------
    psi_r = np.asarray(GIXOS["Psi_R"], dtype=float)

    # ------------------------------------------------------------
    # build Psi_R legend label from resolution metadata
    # ------------------------------------------------------------
    if pr.get("resolution_mode", None) == 0:
        res_str = "circular"
    elif pr.get("resolution_mode", None) == 1:
        res_str = "slit"
    else:
        res_str = "resolution ?"

    bkg_mode = pr.get("bkg_mode", None)
    if bkg_mode is None:
        bkg_str = "w/o bkg"
    else:
        bkg_str = "with bkg"

    # sigma_CW at Qz = 0.6 A^-1
    sigma_str = None
    if "sigma_CW" in GIXOS and GIXOS["sigma_CW"] is not None:
        sigma_cw = np.asarray(GIXOS["sigma_CW"], dtype=float).ravel()
        if sigma_cw.size > 0:
            idx_sigma = int(np.argmin(np.abs(qz_refl - 0.6)))
            sigma_str = rf"$\sigma_{{\mathrm{{CW}}}}(0.6)= {sigma_cw[idx_sigma]:.1f}\,\AA$"

    if sigma_str is None:
        legend_title = rf"$\delta Q_{{xy,R}}$:" + f"\n{res_str}, {bkg_str}"
    else:
        legend_title = rf"$\delta Q_{{xy,R}}$:" + f"\n{res_str}, {bkg_str}\n{sigma_str}"

    psi_label = r"$\Psi_R$"

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # 1) normalized raw GIXOS
    ax.errorbar(
        qz_raw, y_raw_norm, yerr=err_raw_norm,
        fmt="o", markersize=4,
        color="r", ecolor="r",
        elinewidth=1, capsize=2,
        label=r"GIXOS (norm.)",
        zorder=2
    )

    # 2) structure factor
    ax.errorbar(
        qz_sf, y_sf, yerr=err_sf,
        fmt="o", markersize=4,
        color="k", ecolor="k",
        elinewidth=1, capsize=2,
        label=r"$|\Phi(Q_z)|^2$",
        zorder=3
    )

    # 3) pseudo-reflectivity / Fresnel
    ax.errorbar(
        qz_refl, y_refl_norm, yerr=err_refl_norm,
        fmt="o", markersize=4,
        color="b", ecolor="b",
        elinewidth=1, capsize=2,
        label=r"$R/R_F$",
        zorder=4
    )

    # 4) Psi_R
    ax.plot(
        qz_refl, psi_r,
        linestyle="-", linewidth=1.8,
        color="b",
        label=psi_label,
        zorder=1
    )

    # eta = 2 line
    if "Qz_eta2" in GIXOS and GIXOS["Qz_eta2"] is not None:
        qz_eta2 = float(GIXOS["Qz_eta2"])

        ax.axvline(
            qz_eta2,
            linestyle="--",
            color="k",
            linewidth=1.2
        )

        # annotate eta = 2 slightly left of the dashed line near the top
        ax.text(
            qz_eta2 * 0.98, 0.95,
            r"$\eta = 2$",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=10,
            color="k"
        )
        
    
    
    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_z\,(\AA^{-1})$")
    ax.set_ylabel("intensity")

    # ------------------------------------------------------------
    # build title with metadata
    # ------------------------------------------------------------
    title_lines = []
    
    if title is not None:
        title_lines.append(title)
    
    # scan id
    meas = metadata.get("measurements", None)
    if meas is not None:
        scan_val = meas.get("scan", None)
        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            if scan_arr.size == 1:
                scan_str = f"{int(scan_arr[0])}"
            else:
                scan_str = f"{int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
            title_lines.append(f"scan Id = {scan_str}")
    
    # sample parameters
    tension = samp.get("tension", None)
    temperature = samp.get("temperature", None)
    kappa = samp.get("kappa", None)
    
    param_parts = []
    
    if tension is not None:
        param_parts.append(rf"$\gamma={float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
    if temperature is not None:
        param_parts.append(rf"$T={float(temperature):.1f}\,\mathrm{{K}}$")
    if kappa is not None:
        param_parts.append(rf"$\kappa={float(kappa):.0f}\,k_{{\mathrm{{B}}}}T$")
    
    if len(param_parts) > 0:
        title_lines.append(", ".join(param_parts))
    
    if len(title_lines) > 0:
        ax.set_title("\n".join(title_lines))
    else:
        ax.set_title("Pseudo-reflectivity / roughness-factor analysis")


    # -------------------------------------------------------------------------
    # final
    # -------------------------------------------------------------------------

    ax.legend(title=legend_title, title_fontsize=10)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax




def GIXOS_R_plot(
    GIXOS,
    *,
    metadata=None,
    title=None,
    show=True
):
    """
    Plot reflectivity-style quantities from GIXOS2R output.

    This function visualizes, on a semi-log plot (log10 intensity vs linear Qz):

    - structure factor multiplied by Fresnel reflectivity: SF * RF
    - pseudo-reflectivity R
    - specular roughness factor multiplied by Fresnel reflectivity: Psi_R * RF

    A vertical dashed line is added at the Qz position where eta = 2.

    Parameters
    ----------
    GIXOS : dict
        Output dictionary from `GIXOS2R()`.

    metadata : dict, optional
        Metadata dictionary. If None, `GIXOS["metadata"]` is used.

    title : str, optional
        Plot title. If None, a metadata-based title is generated.

    show : bool, optional
        If True, display the figure. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    ax : matplotlib.axes.Axes
        The axes object used for the plot.
    """
    if metadata is None:
        metadata = GIXOS.get("metadata", None)

    if metadata is None:
        raise ValueError("metadata is required either explicitly or in GIXOS['metadata'].")

    if "PseudoR" not in metadata or metadata["PseudoR"] is None:
        raise ValueError("metadata['PseudoR'] is required.")
    if "sample_params" not in metadata or metadata["sample_params"] is None:
        raise ValueError("metadata['sample_params'] is required.")

    pr = metadata["PseudoR"]
    samp = metadata["sample_params"]

    if "qxy0_select_idx" not in pr or pr["qxy0_select_idx"] is None:
        raise ValueError("metadata['PseudoR']['qxy0_select_idx'] is required.")

    # ------------------------------------------------------------
    # data
    # ------------------------------------------------------------
    qz_sf = np.asarray(GIXOS["SF"][:, 0], dtype=float)
    y_sf_rf = np.asarray(GIXOS["SF"][:, 1], dtype=float) * np.asarray(GIXOS["fresnel"][:, 1], dtype=float)
    err_sf_rf = np.asarray(GIXOS["SF"][:, 2], dtype=float) * np.asarray(GIXOS["fresnel"][:, 1], dtype=float)

    qz_refl = np.asarray(GIXOS["refl"][:, 0], dtype=float)
    y_refl = np.asarray(GIXOS["refl"][:, 1], dtype=float)
    err_refl = np.asarray(GIXOS["refl"][:, 2], dtype=float)

    fresnel = np.asarray(GIXOS["fresnel"][:, 1], dtype=float)
    psi_r = np.asarray(GIXOS["Psi_R"], dtype=float)
    y_psi_rf = psi_r * fresnel

    # ------------------------------------------------------------
    # legend title from resolution metadata
    # ------------------------------------------------------------
    if pr.get("resolution_mode", None) == 0:
        res_str = "circular"
    elif pr.get("resolution_mode", None) == 1:
        res_str = "slit"
    else:
        res_str = "resolution ?"

    bkg_mode = pr.get("bkg_mode", None)
    if bkg_mode is None:
        bkg_str = "w/o bkg"
    else:
        bkg_str = "with bkg"

    sigma_str = None
    if "sigma_CW" in GIXOS and GIXOS["sigma_CW"] is not None:
        sigma_cw = np.asarray(GIXOS["sigma_CW"], dtype=float).ravel()
        if sigma_cw.size > 0:
            idx_sigma = int(np.argmin(np.abs(qz_refl - 0.6)))
            sigma_str = rf"$\sigma_{{\mathrm{{CW}}}}(0.6)= {sigma_cw[idx_sigma]:.1f}\,\AA$"

    if sigma_str is None:
        legend_title = rf"$\delta Q_{{xy,R}}$:" + f"\n{res_str}, {bkg_str}"
    else:
        legend_title = rf"$\delta Q_{{xy,R}}$:" + f"\n{res_str}, {bkg_str}\n{sigma_str}"

    # ------------------------------------------------------------
    # plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    # 1) SF * RF
    ax.errorbar(
        qz_sf, y_sf_rf, yerr=err_sf_rf,
        fmt="o", markersize=4,
        color="k", ecolor="k",
        elinewidth=1, capsize=2,
        label=r"$|\Phi(Q_z)|^2 R_F$",
        zorder=2
    )

    # 2) R
    ax.errorbar(
        qz_refl, y_refl, yerr=err_refl,
        fmt="o", markersize=4,
        color="b", ecolor="b",
        elinewidth=1, capsize=2,
        label=r"$R$",
        zorder=3
    )

    # 3) Psi_R * RF
    ax.plot(
        qz_refl, y_psi_rf,
        linestyle="-", linewidth=1.8,
        color="b",
        label=r"$\Psi_R R_F$",
        zorder=1
    )

    # eta = 2 line and label
    if "Qz_eta2" in GIXOS and GIXOS["Qz_eta2"] is not None:
        qz_eta2 = float(GIXOS["Qz_eta2"])

        ax.axvline(
            qz_eta2,
            linestyle="--",
            color="k",
            linewidth=1.2
        )

        ax.text(
            qz_eta2 * 0.98, 0.95,
            r"$\eta = 2$",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=10,
            color="k"
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$Q_z\,(\AA^{-1})$")
    ax.set_ylabel("intensity")

    # ------------------------------------------------------------
    # title with metadata
    # ------------------------------------------------------------
    title_lines = []

    if title is not None:
        title_lines.append(title)

    meas = metadata.get("measurements", None)
    if meas is not None:
        scan_val = meas.get("scan", None)
        if scan_val is not None:
            scan_arr = np.asarray(scan_val).ravel()
            if scan_arr.size == 1:
                scan_str = f"{int(scan_arr[0])}"
            else:
                scan_str = f"{int(np.min(scan_arr))} - {int(np.max(scan_arr))}"
            title_lines.append(f"scan Id = {scan_str}")

    tension = samp.get("tension", None)
    temperature = samp.get("temperature", None)
    kappa = samp.get("kappa", None)

    param_parts = []
    if tension is not None:
        param_parts.append(rf"$\gamma={float(tension)*1000:.1f}\,\mathrm{{mN/m}}$")
    if temperature is not None:
        param_parts.append(rf"$T={float(temperature):.1f}\,\mathrm{{K}}$")
    if kappa is not None:
        param_parts.append(rf"$\kappa={float(kappa):.0f}\,k_{{\mathrm{{B}}}}T$")

    if len(param_parts) > 0:
        title_lines.append(", ".join(param_parts))

    if len(title_lines) > 0:
        ax.set_title("\n".join(title_lines))
    else:
        ax.set_title("Reflectivity analysis")

    ax.legend(title=legend_title, title_fontsize=10)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax