#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
requires ruamel!
helping function:
- GetScans: get information about the beamtime data, such as a list of scans
- GetYamlConfig: reading yaml configuration file for data processing
'''
import glob, os
from ruamel.yaml import YAML

def GetScans(beamtimefolder='/gpfs/current', scanidx_min=None, scanidx_max=None):
    '''
    giving beamtime folder to get all the scans from the beamtime or within the range
    scanidx_min and scanidx_max specify the range of the scan idx (inclusive)
    input with default value: beamtimefolder = '/gpfs/current', scanidx_min=None, scanidx_max=None
    output:
    scanname_lst, scanid_lst
    '''
    name_lst = []
    scan_lst = []
    fio_set = sorted(glob.glob(beamtimefolder+'/raw/*_*.fio'), key=os.path.getctime)
    #fio_set = sorted(glob.glob(beamtimefolder+'/raw/*_*.fio'), key=os.path.basename)
    for address in fio_set:
        name_lst.append(address)
        scan_lst.append(int(address[-9:-4]))
    
    # Determine effective bounds
    if scanidx_min is None or scanidx_min < 1:
        scanidx_min = min(scan_lst)
    if scanidx_max is None or scanidx_max > max(scan_lst):
        scanidx_max = max(scan_lst)
    
    scan_lst, name_lst = zip(*[
    (i, n) for i, n in zip(scan_lst, name_lst)
    if scanidx_min <= i <= scanidx_max
    ])
    
    return name_lst, scan_lst
    
def GetYamlConfig(yaml_path: str):
    '''
    get the yaml file config for data processing
    input: yaml file path
    return: dictionary according to yaml file
    Caution: if None is needed, leave the input in Yaml file empty, or write null, or ~
    '''
    yaml = YAML(typ='safe')
    # Load YAML
    with open(yaml_path, "r") as f:
        config = yaml.load(f)
    
    return config
    