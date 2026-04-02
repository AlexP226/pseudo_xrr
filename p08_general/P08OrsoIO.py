# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path
import numpy as np

from orsopy import fileio
from orsopy.fileio import Polarization, Reduction, Software

from xray_general_io.OrsoIO import OrsoIO
from p08_general.metadata_reader import load_beamtime_metadata
import p08_general.fio_reader as fio_reader
from p08_general.P08ScanTools import Scan, ScanAnalyzer


class P08OrsoIO(OrsoIO):
    """
    P08-specific ORSO-style file I/O adapter.

    This class extends the general OrsoIO class with:
    - PETRA III / P08 beamtime JSON metadata loading
    - P08 scan file (.fio / .nxs) metadata loading
    - setup-specific mapping for:
        * Langmuir trough GID setup
        * Liquid setup / LISA
        * Kohzu setup
    """

    def __init__(self):
        super().__init__()
        self.p08metadata = {}

    def load_metadata_from_json(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        self.p08metadata = load_beamtime_metadata(filepath)
        return self.p08metadata

    def load_metadata_from_scan(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"scan metadata file not found: {filepath}")

        self.scanmetadata["filepath"] = filepath
        self.scanmetadata["scan_name"] = Path(filepath).stem.rsplit("_", 1)[0]
        self.scanmetadata["scan_no"] = Path(filepath).stem.split("_")[-1]
        (
            self.scanmetadata["motor_positions"],
            self.scanmetadata["column_names"],
            self.scanmetadata["data"],
            self.scanmetadata["header_info"],
        ) = fio_reader.read(filepath)
        return self.scanmetadata

    def populate_metadata_to_header(self):
        """
        Populate generic ORSO header fields from loaded P08 metadata and scan metadata.
        """
        p08md = self.p08metadata or {}
        scanmd = self.scanmetadata or {}
        today = datetime.date.today().isoformat()

        self.header.title = scanmd.get("header_info", {}).get("title", "")
        self.header.DateCreate = today
        self.header.SampleName = scanmd.get("scan_name", "")

        # person
        self.header.DataSource.person["user_name"] = p08md.get("pi", {}).get("lastname", "")
        self.header.DataSource.person["user_affil"] = p08md.get("pi", {}).get("institute", "")
        self.header.DataSource.person["contact"] = p08md.get("pi", {}).get("email", "")
        self.header.DataSource.person["comment"] = (
            "DOOR username: "
            + p08md.get("pi", {}).get("username", "")
            + "; DOOR userId: "
            + p08md.get("pi", {}).get("userId", "")
        )

        # experiment
        self.header.DataSource.experiment["title"] = p08md.get("title", "")
        self.header.DataSource.experiment["instrument"] = (
            "beamline " + p08md.get("beamline", "") + " / " + p08md.get("beamlineSetup", "")
        )
        if p08md.get("eventStart", ""):
            self.header.DataSource.experiment["start_date"] = str(
                datetime.datetime.strptime(
                    p08md.get("eventStart", ""),
                    "%Y-%m-%d %H:%M:%S"
                ).date()
            )
        self.header.DataSource.experiment["facility"] = "DESY/" + p08md.get("facility", "")
        self.header.DataSource.experiment["proposalID"] = p08md.get("proposalId", "")

        # instrument settings defaults
        self.header.DataSource.instrument_settings["incident_angle"] = fileio.base.Value([], unit="unknown")
        self.header.DataSource.instrument_settings["incident_angle"].movement = "unknown"

        energyfmb = scanmd.get("motor_positions", {}).get("energyfmb", [])
        if energyfmb not in ([], None):
            self.header.DataSource.instrument_settings["wavelength"] = fileio.base.Value(
                12400.0 / energyfmb,
                unit="angstrom"
            )

        beamline_setup = p08md.get("beamlineSetup", "")

        if "Langmuir" in beamline_setup:
            self.header.DataSource.instrument_settings["polarization"] = Polarization.sigma
            self.header.DataSource.instrument_settings["configuration"] = (
                "fixed incidence grazing incidence scattering setup - liquid surface "
                "(doi:10.1088/1742-6596/2380/1/012047), "
                "GIXOS/pseudoXRR (doi: 10.1107/S1600576724002887; 10.1103/znt1-fmx6)"
            )
        elif "Liquid" in beamline_setup:
            self.header.DataSource.instrument_settings["polarization"] = Polarization.sigma
            self.header.DataSource.instrument_settings["configuration"] = (
                "double crystal beam tilter - liquid interface "
                "(doi: 10.1107/S1600577513026192), theta-2theta"
            )
        elif "Kohzu" in beamline_setup:
            self.header.DataSource.instrument_settings["configuration"] = (
                "6-circle diffractometer "
                "(doi: 10.1107/S0909049511047236), theta-2theta"
            )
        else:
            self.header.DataSource.instrument_settings["configuration"] = (
                "special setup, see beamline logbook"
            )

        self.header.DataSource.instrument_settings["sample_detector_distance"] = fileio.base.Value([], unit="unknown")
        self.header.DataSource.instrument_settings["roi_specular"] = fileio.base.ValueVector([], [], [], unit="unknown")
        self.header.DataSource.instrument_settings["roi_specular"].definition = "unknown"
        self.header.DataSource.instrument_settings["roi_specular"].configuration = "unknown"
        self.header.DataSource.instrument_settings["roi_specular"].orientation_normal = "unknown"
        self.header.DataSource.instrument_settings["roi_bkg_offspec"] = fileio.base.Value([], unit="unknown", comment="")

        # sample
        self.header.DataSource.sample["name"] = self.header.SampleName
        self.header.DataSource.sample["sample name"] = scanmd.get("header_info", {}).get("sample_name", "")
        self.header.DataSource.sample["composition"] = scanmd.get("header_info", {}).get("chemical_formula", "")
        self.header.DataSource.sample["description"] = scanmd.get("header_info", {}).get("sample_description", "")
        self.header.DataSource.sample["identifier"] = scanmd.get("header_info", {}).get("sample_identifier", "")

        if "Langmuir" in beamline_setup:
            self.header.DataSource.sample["category"] = "vapour/liquid"

        return self.header


    def load_xrr_from_scan(self, detector, roi, xcolName=None, bckroi=None, abs_corr=False):
        """
        Extract specular reflectivity intensity from a scan by integrating a detector ROI.
    
        Parameters
        ----------
        detector : str
            Detector name used by ScanAnalyzer.extract_rois().
    
        roi : list[int]
            Specular ROI in the form [y0, x0, y1, x1].
    
        xcolName : str, optional
            Column or scanned motor name to use as x-axis. If None, use the first
            scanned motor.
    
        bckroi : list[int], optional
            Off-specular background ROI in the form [y0, x0, y1, x1]. If given,
            background is subtracted from the specular ROI intensity.
    
        abs_corr : bool, optional
            If True, apply absorber correction and remove duplicate points before
            extracting the ROI intensities.
    
        Returns
        -------
        np.ndarray
            Two-column array [x, I_R].
        """
        scan = Scan()
        scan.load_scan(self.scanmetadata["filepath"])
    
        # -----------------------------------------------------------------
        # reduction bookkeeping
        # -----------------------------------------------------------------
        corrections = ["ROI(s) integration"]
        call_parts = []
    
        if abs_corr:
            scan.correct_absorber()
            scan.remove_double()
            corrections.extend([
                "absorber correction",
                "remove double points",
            ])
            call_parts.extend([
                "correct_absorber()",
                "remove_double()",
            ])
    
        # -----------------------------------------------------------------
        # choose x column
        # -----------------------------------------------------------------
        xcol_idx = 0
        if xcolName is not None and xcolName in scan.scan_motor_names:
            for idx in range(len(scan.scan_motor_names)):
                if scan.scan_motor_names[idx] == xcolName:
                    xcol_idx = idx
                    break
            xcol = scan.scan_motors[xcol_idx]
            xcol_name = scan.scan_motor_names[xcol_idx]
    
        elif xcolName is not None and xcolName not in scan.scan_motor_names:
            xcol = scan.data[xcolName]
            xcol_name = xcolName
    
        else:
            xcol = scan.scan_motors[xcol_idx]
            xcol_name = scan.scan_motor_names[xcol_idx]
    
        # -----------------------------------------------------------------
        # extract specular ROI
        # -----------------------------------------------------------------
        result = ScanAnalyzer.extract_rois(scan, {"fit_roi": roi})
        roi_intensity = np.array(result[detector]["fit_roi"])
        call_parts.append("extract_rois()")
    
        # -----------------------------------------------------------------
        # optional off-spec background subtraction
        # -----------------------------------------------------------------
        if bckroi is not None:
            result = ScanAnalyzer.extract_rois(scan, {"bck_int": bckroi})
            bck_int = np.array(result[detector]["bck_int"])
            roi_intensity = roi_intensity - bck_int
            corrections.append("off-specular background subtraction")
    
        # -----------------------------------------------------------------
        # x-axis metadata
        # -----------------------------------------------------------------
        if xcol_name in ["om", "om_position", "tt", "tt_position", "omh", "tth", "alpha_pos", "beta_pos"]:
            xcol_unit = "deg"
            xcol_quantity = "angle"
        elif xcol_name in ["q", "qz"]:
            xcol_unit = "1/angstrom"
            xcol_quantity = "wavevector transfer"
        else:
            xcol_unit = None
            xcol_quantity = None
    
        self.header.ColDescription[0] = fileio.base.Column(
            name=xcol_name,
            unit=xcol_unit,
            physical_quantity=xcol_quantity
        )
        self.header.ColDescription[1] = fileio.base.Column(
            name="I_R",
            unit=None,
            physical_quantity="reflection intensity"
        )
    
        # -----------------------------------------------------------------
        # populate ROI metadata
        # ROI format assumed as [y0, x0, y1, x1]
        # -----------------------------------------------------------------
        roi_y = roi[2] - roi[0] + 1
        roi_x = roi[3] - roi[1] + 1
        roi_unit = f"{detector} px"
    
        self.header.DataSource.instrument_settings["roi_specular"] = fileio.base.ValueVector(
            roi_x, roi_y, 0, unit=roi_unit
        )
        self.header.DataSource.instrument_settings["roi_specular"].definition = "FW"
        self.header.DataSource.instrument_settings["roi_specular"].configuration = "rectangular"
        self.header.DataSource.instrument_settings["roi_specular"].orientation_normal = "unknown"
        self.header.DataSource.instrument_settings["roi_specular"].comment = (
           f"ROI = [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}] (y0, x0, y1, x1); detector rotation in xy is unknown"
        )
    
        if bckroi is not None:
            delta_x = roi[1] - bckroi[1]
            delta_y = roi[0] - bckroi[0]
    
            self.header.DataSource.instrument_settings["roi_bkg_offspec"] = fileio.base.ValueVector(
                delta_x, delta_y, 0, unit=roi_unit
            )
            self.header.DataSource.instrument_settings["roi_bkg_offspec"].use = True
            self.header.DataSource.instrument_settings["roi_bkg_offspec"].configuration = "rectangular"
            self.header.DataSource.instrument_settings["roi_bkg_offspec"].comment = (
                "offset of off-specular background ROI relative to specular ROI; detector rotation in xy is unknown"
            )
        else:
            self.header.DataSource.instrument_settings["roi_bkg_offspec"] = fileio.base.Value(
                [], unit=roi_unit
            )
            self.header.DataSource.instrument_settings["roi_bkg_offspec"].use = False
    
        # -----------------------------------------------------------------
        # reduction metadata
        # -----------------------------------------------------------------
        if len(corrections) == 0:
            corrections = ["specular ROI integration"]
    
        call_str = "; ".join(call_parts)
    
        self.header.DataReduction = Reduction(
            software=[Software(name="p08_general.P08OrsoIO")],
            corrections=corrections,
            call=call_str,
            comment=None,
        )
    
        # -----------------------------------------------------------------
        # output dataset
        # -----------------------------------------------------------------
        self.Dataset = np.array([xcol, roi_intensity]).T
        return self.Dataset