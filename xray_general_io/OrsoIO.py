# -*- coding: utf-8 -*-
"""
General ORSO-style file I/O utilities for x-ray experiments.

This module is facility-independent. It provides:
- a generic header/container structure
- conversion of that structure into orsopy objects
- ORSO dataset creation and saving

Facility- or beamline-specific metadata loading should be implemented in
separate adapter classes, e.g. P08OrsoIO, that inherit from OrsoIO.
"""

import datetime
import numpy as np

from orsopy import fileio
from orsopy.fileio import (
    Reduction, Person, Orso, Experiment,
    Sample, DataSource, Measurement, InstrumentSettings
)


def _create_header_class():
    """
    Internal factory: defines a Header class for generic OrsoIO.
    """
    class HeaderDataSource:
        """
        Container for data source details: owner, experiment, sample, measurements.
        """
        def __init__(self):
            self.person = {
                "user_name": "",
                "user_affil": "",
                "contact": "",
                "comment": "",
            }

            self.experiment = {
                "title": "",
                "instrument": "",
                "start_date": "",
                "probe": "x-ray",
                "facility": "",
                "proposalID": "",
            }

            self.sample = {
                "name": "",
                "category": "",
                "composition": "",
                "description": "",
            }

            self.instrument_settings = {
                "incident_angle": [],
                "wavelength": [],
                "polarization": [],
                "configuration": "",
                "sample_detector_distance": [],
                "roi_specular": [],
                "roi_bkg_offspec": [],
                "comment": "",
            }

            self.measurement = {
                "instrument_settings": self.instrument_settings,
                "data_files": [],
                "additional_files": [],
                "scheme": "angle-dispersive",
                "comment": "",
            }

        def __repr__(self):
            return (
                f"HeaderDataSource(person={self.person!r}, "
                f"experiment={self.experiment!r}, "
                f"sample={self.sample!r}, "
                f"instrument_settings={self.instrument_settings!r}, "
                f"measurement={self.measurement!r})"
            )

    class Header:
        """
        Container for ORSO header fields.
        """
        def __init__(self):
            self.title = ""
            self.DateCreate = ""
            self.SampleName = ""
            self.DataType = ""
            self.DataSource = HeaderDataSource()
            self.DataReduction = "unknown"
            self.ColDescription = fileio.orso.Orso.empty().columns
            self.Dataset = []

        def __repr__(self):
            return (
                f"Header(title={self.title!r}, DateCreate={self.DateCreate!r}, "
                f"SampleName={self.SampleName!r}, DataType={self.DataType!r}, "
                f"DataSource={self.DataSource!r}, "
                f"DataReduction={self.DataReduction!r}, "
                f"ColDescription={self.ColDescription!r}, Dataset={self.Dataset!r})"
            )

    return Header


class OrsoIO:
    """
    General ORSO-style file I/O handler.

    This class is facility-independent. It does not assume any specific
    metadata source or beamline file structure.
    """

    def __init__(self):
        Header = _create_header_class()
        self.header = Header()

        # generic metadata containers
        self.external_metadata = {}
        self.scanmetadata = {
            "filepath": None,
            "scan_name": None,
            "scan_no": None,
            "motor_positions": [],
            "column_names": [],
            "data": [],
            "header_info": [],
        }

        self.Dataset = np.full((1, 2), np.nan)

    def set_basic_header(
        self,
        *,
        title="",
        sample_name="",
        data_type="",
        date_create=None,
    ):
        """
        Set basic top-level header fields.
        """
        self.header.title = title
        self.header.SampleName = sample_name
        self.header.DataType = data_type
        self.header.DateCreate = (
            date_create if date_create is not None else datetime.date.today().isoformat()
        )

    def populate_metadata_to_orsodatasource(self):
        """
        Convert the internal header dictionaries into orsopy DataSource objects.
        """
        person = Person(
            self.header.DataSource.person["user_name"],
            self.header.DataSource.person["user_affil"],
        )
        for key, val in self.header.DataSource.person.items():
            if key in ("user_name", "user_affil"):
                continue
            setattr(person, key, val)

        experiment = Experiment(
            self.header.DataSource.experiment["title"],
            self.header.DataSource.experiment["instrument"],
            self.header.DataSource.experiment["start_date"],
            self.header.DataSource.experiment["probe"],
        )
        for key, val in self.header.DataSource.experiment.items():
            if key in ("title", "instrument", "start_date", "probe"):
                continue
            setattr(experiment, key, val)

        instrument_settings = InstrumentSettings(
            self.header.DataSource.instrument_settings["incident_angle"],
            self.header.DataSource.instrument_settings["wavelength"],
        )
        for key, val in self.header.DataSource.instrument_settings.items():
            if key in ("incident_angle", "wavelength"):
                continue
            setattr(instrument_settings, key, val)

        measurement = Measurement(
            instrument_settings,
            [self.scanmetadata["scan_no"]],
        )
        for key, val in self.header.DataSource.measurement.items():
            if key in ("instrument_settings", "data_files"):
                continue
            setattr(measurement, key, val)

        sample = Sample(self.header.DataSource.sample["name"])
        for key, val in self.header.DataSource.sample.items():
            if key == "name":
                continue
            setattr(sample, key, val)

        self.header.OrsoDataSource = DataSource(
            person, experiment, sample, measurement
        )
        return self.header.OrsoDataSource

    def create_orsodataset(self, exportpath=None):
        """
        Create an ORSO dataset and optionally save it.
        """
        if isinstance(self.header.DataReduction, Reduction):
            reduction_use = self.header.DataReduction
        else:
            reduction_use = Reduction(self.header.DataReduction)

        orso_class = Orso(
            self.header.OrsoDataSource,
            reduction=reduction_use,
            columns=self.header.ColDescription,
        )
        self.orsodataset = fileio.orso.OrsoDataset(
            info=orso_class,
            data=self.Dataset
        )

        if exportpath:
            fileheader = (
                self.header.title + " | "
                + self.header.DateCreate + " | "
                + self.header.SampleName + " | "
                + self.header.DataType
            )
            fileio.orso.save_orso(
                datasets=[self.orsodataset],
                fname=exportpath,
                comment=fileheader
            )

        return self.orsodataset