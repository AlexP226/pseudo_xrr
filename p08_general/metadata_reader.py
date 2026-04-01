# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:13:53 2025

@author: shenc
import JSON metadata file of the beamtime from PETRA III facility into a dictonary
proposalId is defined as the DOOR_proposalId.beamtimeId to be consistent with DESY-SciCat decision
"""
import json
import os
import argparse

def load_beamtime_metadata(filepath):
    """
    Load the JSON file, rename the original `proposalId` to `DOOR_proposalId`,
    and set `proposalId` to "<beamtimeId>.<original proposalId>".
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filename = os.path.basename(filepath).lower()
    if "commissioning" in filename:
        # Use "id" for both fields
        id_val = data.get('id')
        data['DOOR_proposalId'] = id_val
        data['proposalId'] = id_val if id_val is not None else ""
    else:
        # Preserve original proposalId, then combine with beamtimeId
        orig_pid = data.get('proposalId')
        beamtime_id = data.get('beamtimeId')

        data['DOOR_proposalId'] = orig_pid
        if beamtime_id and orig_pid:
            data['proposalId'] = f"{orig_pid}.{beamtime_id}"
        else:
            data['proposalId'] = ""
        data['filepath'] = filepath    
    return data

# if __name__ == "__main__":
#     json_path = "U:/p08/2024/data/11020531/beamtime-metadata-11020531.json"
#     transformed = load_and_transform(json_path)
#     print(json.dumps(transformed, indent=2))

def main():
    purpose = (
        "Import JSON metadata file of the beamtime from PETRA III facility into a dictionary.\n"
        "The entry `proposalId` is defined as `<beamtimeId>.<original proposalId>`\n"
        "(stored as `DOOR_proposalId` + `beamtimeId`) to be consistent with DESY‑SciCat decision."
    )

    parser = argparse.ArgumentParser(
        description=purpose,
        epilog=(
            "Example:\n"
            "  python transform_metadata.py /mnt/data/beamtime-metadata-11020531.json\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'filepath',
        help="Path to the JSON metadata file from PETRA III facility"
    )
    args = parser.parse_args()

    transformed = load_beamtime_metadata(args.filepath)
    print(json.dumps(transformed, indent=2))

if __name__ == "__main__":
    main()