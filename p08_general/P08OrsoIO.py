# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:24:22 2025

@author: shenc
Class-based interface for ORSO-style file I/O at the P08 beamline.
Provides methods to load beamtime metadata using metadata_reader and
placeholder methods for various ORSO output formats.

orsopy version 1.2.2 (2025-06-27)
orsopy.fileio contains these modules, to prepare metadata in the way for output:
    - Reduction(software[str])
    - Person(user_name, user_affil), passed to "user"
    - Experiment(title, instrument, start_date, probe, facility, proposalID, doi), passed to "experiment"
    - Sample(sample_name[str],category=str,composition=str,description=str,size=valueVector,environment=StrList,sample_parameters=dictionary), passed to "sample"
    - InstrumentSettings(angle[union], wavelength[union], polarization=union,configuration=str,comment=None)
    - Measurement(instrument_settings, data_files,additional_files=StrList,scheme=Str,comment=str), data_files are a list of strings for raw data, passed to measurement
    - DataSource(user, experiment, sample, measurement), passed as data_source_info
    fileio.base.Value, ValueRange and ValueVector can be used to set more metadata that are not included in the above functions
    Further objects can be added after defining those metadata
    then also 
    - Orso(data_source_info, reduction=..., columns=...) (columns is for information of the columns)

"""

import os, glob
import datetime
from pathlib import Path
import numpy as np
# beamtime metadata is loaded from json of gpfs
from p08_general.metadata_reader import load_beamtime_metadata
import p08_general.fio_reader as fio_reader
from p08_general.P08ScanTools import Scan, ScanAnalyzer
from orsopy import fileio
from orsopy.fileio import (Reduction, Person, Orso, Experiment, Polarization,
                           Sample, DataSource, Measurement, InstrumentSettings)

def _create_header_class():
    """
    Internal factory: defines a Header class for P08OrsoIO.
    """
    class DataSource:
        """
        Container for data source details: owner, experiment, sample, measurements.
        """
        def __init__(self):
            '''
            self.person = PersonInfo()
            self.experiment = ExperimentInfo()
            self.sample = SampleInfo()
            self.instrument_settings = InstrumentSettingsInfo()
            self.measurement = MeasurementInfo()
            '''
            self.person = {'user_name': "", "user_affil": "", "contact": "", "comment": ""}
            self.experiment = {
                                "title": "",
                                "instrument": "",
                                "start_date": "",
                                "probe": "x-ray",
                                "facility": "DESY",
                                "proposalID": "",
                                }
            self.sample = {
                            "name": "", # fetch from fio file name
                            "category": "", # beamside/subphase, liquid/solid, vapour/liquid, liquid/liquid, vacuum/solid
                            "composition": "", # fetch from chemical_formula
                            "description": "", # fetch from sample_name 
                            }
            self.instrument_settings = {
                                            "incident_angle": [],           #"magnitude": [], "range": [], "unit": "", "movement": "",
                                            "wavelength": [],               #"magnitude": [], "unit": "", fetch from fio
                                            "polarization": [],             # on Langmuir GID and LISA, sigma_sigma, on Kohzu sample dependent
                                            "configuration": "",
                                            "sample_detector_distance": [], #"magnitude": [], "unit": "",
                                            "roi_specular" : [],            #"configuration": "", "magnitude": [], "unit": "", "definition": "", "comment": "", region of interest of specular reflection integration
                                            "roi_bkg_offspec": [],          #"use": [], "magnitute": [], "unit": "", "comment": "", position of the region of interest centre of the offspecular bkg from the specular position
                                            "comment": ""
                                        }
            self.measurement = {
                                "instrument_settings": self.instrument_settings,
                                "data_files": [],
                                "additional_files": [],
                                "scheme": "angle-dispersive"    # angle- and energy-dispersive, angle dispersive, energy dispersive
                                }

        def __repr__(self):
            return (
                f"DataSource(person={self.person!r}, experiment={self.experiment!r}, "
                f"sample={self.sample!r}, instrument_settings={self.instrument_settings!r}, "
                f"measurement={self.measurement!r})"
            )

    
    class Header:
        """
        Container for header fields for ORSO file
        """
        def __init__(self):
            # Initialize all fields empty
            self.title = ""                 # title of the measurement from fio
            self.DateCreate = ""            # date when creating the ORSO file
            self.SampleName = ""            # sample name from fio
            self.DataType = ""              # R(')(Qz), PseudoR(')(Qz), R*(Qz)|2theta, |Phi(Qz)|^2, Psi(Qz)
            self.DataSource     = DataSource()
            self.DataReduction = "unknown"
            # define column
            self.ColDescription = fileio.orso.Orso.empty().columns
            self.Dataset = []

        def __repr__(self):
            return (
                f"Header(title={self.title!r}, DateCreate={self.DateCreate!r}, "
                f"SampleName={self.SampleName!r}, DataType={self.DataType!r}, "
                f"DataSource={self.DataSource!r}, DataReduction={self.DataReduction!r}, "
                f"ColDescription={self.ColDescription!r}, Dataset={self.Dataset!r})"
                
            )
    return Header


class P08OrsoIO:
    """
    ORSO-style file I/O handler for P08 beamline data.

    Attributes:
        p08metadata (dict): Dictionary of beamtime metadata loaded from JSON.
    """

    def __init__(self):
        """
        Initialize a new P08OrsoIO instance with empty metadata.
        """
        Header = _create_header_class()
        self.header = Header()
        # P08 metadata is for the json file
        self.p08metadata = {}
        # load the fio file content
        self.scanmetadata= {'scan_name': None,
                            'scan_no': None,
                            'motor_positions':[], 
                            'column_names': [], 
                            'data': [], # this is the data field from the fio file
                            'header_info': []}
        self.Dataset= np.full((1, 2), np.nan)
        
        
    def load_metadata_from_json(self, filepath):
        """
        Load beamtime metadata from a JSON file into `p08metadata`.

        Args:
            filepath (str): Path to the json file file.

        Returns:
            dict: The loaded metadata dictionary.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")

        self.p08metadata = load_beamtime_metadata(filepath)
        #self.populate_header()
        return self.p08metadata
        
    def load_metadata_from_scan(self, filepath):
        """
        Giving the scan meta file path (fio or nxs)
        Load scan metadata into `scanmetadata`.

        Args:
            filepath (str): Path to the scan file file.

        Returns:
            dict: The loaded metadata dictionary.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"scan metadata file not found: {filepath}")
        self.scanmetadata['filepath'] = filepath
        self.scanmetadata['scan_name'] = Path(filepath).stem.rsplit('_', 1)[0]
        self.scanmetadata['scan_no'] = Path(filepath).stem.split('_')[-1]
        self.scanmetadata['motor_positions'], self.scanmetadata['column_names'], self.scanmetadata['data'], self.scanmetadata['header_info'] = fio_reader.read(filepath)
        return self.scanmetadata
        
    def populate_metadata_to_header(self):
        """
        Populate `self.header` fields from loaded p08metadata and scanmetadata:
            - header with (this goes into the 2nd line)
                - title         <= from .fio
                - DateCreate    <= when creating the orso file
                - SampleName    <= from the name of the .fio
                - DataType          <= R(')(Qz), PseudoR(')(Qz), R*(Qz)|2theta, |Phi(Qz)|^2, Psi(Qz)
            - header.DataSource
        """
        p08md = self.p08metadata or {}
        scanmd = self.scanmetadata or {}
        today = datetime.date.today().isoformat()
        self.header.title = scanmd.get("header_info",{}).get("title","")
        self.header.DateCreate  = today
        self.header.SampleName = scanmd.get("scan_name","")
        
        # DataSource
        # get person information
        self.header.DataSource.person['user_name'] = p08md.get("pi", {}).get("lastname", "")
        self.header.DataSource.person['user_affil'] = p08md.get("pi", {}).get("institute", "")
        self.header.DataSource.person['contact'] = p08md.get("pi", {}).get("email", "")
        self.header.DataSource.person['comment'] = f'DOOR username: '+p08md.get("pi", {}).get("username", "")+f'; DOOR userId: '+p08md.get("pi", {}).get("userId", "")
        # experiment metadata
        self.header.DataSource.experiment['title'] = p08md.get('title','')
        self.header.DataSource.experiment['instrument'] = f'beamline ' + p08md.get('beamline','') + f' / '+p08md.get('beamlineSetup','') 
        self.header.DataSource.experiment['start_date'] = str((datetime.datetime.strptime(p08md.get('eventStart',''), "%Y-%m-%d %H:%M:%S")).date())
        self.header.DataSource.experiment['facility'] = self.header.DataSource.experiment['facility'] + '/' + p08md.get('facility','')
        self.header.DataSource.experiment['proposalID'] = p08md.get('proposalId','')
        # instrument settings           
        self.header.DataSource.instrument_settings['incident_angle'] = fileio.base.Value([],unit='unknown')
        self.header.DataSource.instrument_settings['incident_angle'].movement = "unknown"
        energyfmb = scanmd.get("motor_positions",{}).get("energyfmb",[])
        self.header.DataSource.instrument_settings['wavelength'] = fileio.base.Value(12400.0/energyfmb, unit ="angstrom")
        print(p08md.get('beamlineSetup',''))
        if 'Langmuir' in p08md.get('beamlineSetup',''):
            self.header.DataSource.instrument_settings['polarization']=Polarization.sigma
            self.header.DataSource.instrument_settings['configuration']="fixed incidence grazing incidence scattering setup - liquid surface (doi:10.1088/1742-6596/2380/1/012047), GIXOS/pseudoXRR (doi: 10.1107/S1600576724002887; 10.1103/znt1-fmx6)"
        elif 'Liquid' in p08md.get('beamlineSetup',''):
            self.header.DataSource.instrument_settings['polarization']=Polarization.sigma
            self.header.DataSource.instrument_settings['configuration']="double crystal beam tilter - liquid interface (doi: 10.1107/S1600577513026192), theta-2theta"
        elif 'Kohzu' in p08md.get('beamlineSetup',''):
            self.header.DataSource.instrument_settings['configuration']="6-circle diffractometer (doi: 10.1107/S0909049511047236), theta-2theta"
        else:
            self.header.DataSource.instrument_settings['configuration']="special setup, see p08 logbook"
        self.header.DataSource.instrument_settings['sample_detector_distance'] = fileio.base.Value([], unit ="unknown")
        self.header.DataSource.instrument_settings['roi_specular'] = fileio.base.ValueVector([],[],[], unit ="unknown")
        self.header.DataSource.instrument_settings['roi_specular'].definition = "unknown"
        self.header.DataSource.instrument_settings['roi_specular'].configuration="unknown"
        self.header.DataSource.instrument_settings['roi_specular'].orientation_normal="unknown"
        self.header.DataSource.instrument_settings['roi_bkg_offspec'] = fileio.base.Value([], unit ="unknown", comment="")
        # sample metadata
        self.header.DataSource.sample["name"] = self.header.SampleName
        self.header.DataSource.sample["sample name"] = scanmd.get("header_info",{}).get("sample_name","")
        self.header.DataSource.sample["composition"] = scanmd.get("header_info",{}).get("chemical_formula","")
        self.header.DataSource.sample["description"] = scanmd.get("header_info",{}).get("sample_description","")
        self.header.DataSource.sample["identifier"] = scanmd.get("header_info",{}).get("sample_identifier","")
        if 'Langmuir' in p08md.get('beamlineSetup',''):
            self.header.DataSource.sample["category"] = "vapour/liquid"
        return self.header
        
    def populate_metadata_to_orsodatasource(self):
        """
        populate the metadata entry into orsodatasource type: Value, ValueRange or ValueVector class
        Person() default: user_name, user_affil
        Experiment() default: title, instrument, start_date, probe
        InstrumentSettings() default: incident_angle['magnitude'], wavelength[magnitude]
        Measurement(): instrument_settings, data_files
        Sample(): name
        All the other are optional and can be added / maybe added later
        """
        person = Person(self.header.DataSource.person['user_name'], self.header.DataSource.person['user_affil'])
        for key, val in self.header.DataSource.person.items():
            if key in ('user_name', 'user_affil'):
                continue
            setattr(person, key, val)
        experiment = Experiment(self.header.DataSource.experiment['title'], self.header.DataSource.experiment['instrument'],self.header.DataSource.experiment['start_date'],self.header.DataSource.experiment['probe'])
        for key, val in self.header.DataSource.experiment.items():
            if key in ('title', 'instrument','start_date','probe'):
                continue
            setattr(experiment, key, val)
        instrument_settings = InstrumentSettings(self.header.DataSource.instrument_settings['incident_angle'],self.header.DataSource.instrument_settings['wavelength'])
        for key, val in self.header.DataSource.instrument_settings.items():
            if key in ('incident_angle', 'wavelength'):
                continue
            setattr(instrument_settings, key, val)        
        measurement = Measurement(instrument_settings, [self.scanmetadata['scan_no']])
        for key, val in self.header.DataSource.measurement.items():
            print(key)
            if key in ('instrument_settings', 'data_files'):
                continue
            setattr(measurement, key, val)           
        sample = Sample(self.header.DataSource.sample["name"])
        for key, val in self.header.DataSource.sample.items():
            if key in ('name'):
                continue
            setattr(sample, key, val)  
        
        self.header.OrsoDataSource = DataSource(person, experiment, sample, measurement)
        return self.header.OrsoDataSource
    
    # create orso dataset
    def create_orsodataset(self, exportpath = None):
        """
        create an orsodataset for direct orso output
        
        """
        #orsometadata = self.populate_metadata_to_orsodatasource()
        if isinstance(self.header.DataReduction, Reduction):
            reduction_use = self.header.DataReduction
        else:
            reduction_use = Reduction(self.header.DataReduction)
        orso_class = Orso(self.header.OrsoDataSource, reduction=reduction_use, columns=self.header.ColDescription)
        self.orsodataset = fileio.orso.OrsoDataset(info=orso_class, data=self.Dataset)
        if exportpath:
            fileheader = self.header.title + ' | ' + self.header.DateCreate + ' | ' + self.header.SampleName + ' | ' + self.header.DataType
            fileio.orso.save_orso(datasets=[self.orsodataset], fname=exportpath, comment = fileheader)  # note that the first input is a list of datasets
        return self.orsodataset
    
    def load_xrr_from_scan(self, detector, roi, xcolName=None, bckroi=None, abs_corr=False):
        scan = Scan()
        scan.load_scan(self.scanmetadata['filepath'])
        if abs_corr:
            scan.correct_absorber()
            scan.remove_double()
        
        xcol_idx = 0
        if not xcolName is None and xcolName in scan.scan_motor_names:
            for idx in range(len(scan.scan_motor_names)):
                if scan.scan_motor_names[idx] == xcolName:
                    xcol_idx = idx
                    break                  
            xcol = scan.scan_motors[xcol_idx]
            xcol_name = scan.scan_motor_names[xcol_idx]
        elif not xcolName is None and not xcolName in scan.scan_motor_names:
            xcol = scan.data[xcolName]
            xcol_name = xcolName           
        else:
            xcol = scan.scan_motors[xcol_idx]
            xcol_name = scan.scan_motor_names[xcol_idx]
            
        result = ScanAnalyzer.extract_rois(scan, {"fit_roi" : roi})
        roi_intensity = np.array(result[detector]["fit_roi"])
        if not bckroi is None:
            result = ScanAnalyzer.extract_rois(scan, {"bck_int" : bckroi})
            bck_int = np.array(result[detector]["bck_int"])
            roi_intensity =roi_intensity - bck_int
        
        if xcol_name in ['om', 'om_position', 'tt', 'tt_position', 'omh', 'tth', 'alpha_pos', 'beta_pos']:
            xcol_unit = 'deg'
            xcol_quantity = 'angle'
        elif xcol_name in ['q', 'qz']:
            xcol_unit = '1/angstrom'
            xcol_quantity = 'wavevector transfer'
        else:
            xcol_unit = None
            xcol_quantity = None
        
        self.header.ColDescription[0] = fileio.base.Column(name=xcol_name, unit=xcol_unit, physical_quantity=xcol_quantity)
        self.header.ColDescription[1] = fileio.base.Column(name='I_R', unit=None, physical_quantity='reflection intensity')

        self.Dataset = np.array([xcol, roi_intensity]).T
        
        return self.Dataset
    
    def summary(self):
        """
        Print a summary of the loaded metadata fields.

        Returns:
            str: Summary string detailing key metadata entries.
        """
        if not self.p08metadata:
            return "No metadata loaded."

        keys = list(self.p08metadata.keys())
        summary_lines = [f"Loaded metadata with {len(keys)} fields:"]
        for key in keys:
            summary_lines.append(f"  - {key}: {self.p08metadata[key]}")
        return "\n".join(summary_lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="P08OrsoIO: Load beamtime metadata and provide ORSO I/O utilities for P08."
    )
    parser.add_argument(
        'metadata_file', help='Path to the beamtime metadata JSON file'
    )
    args = parser.parse_args()

    io = P08OrsoIO()
    p08metadata = io.load_metadata(args.metadata_file)
    print(io.summary())
