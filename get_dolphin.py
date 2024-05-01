# OUTDATED: used for parsing the slippi public dataset (which is not suitable for biometrics)
# exists to sort emulator replays from console replays

from slippi.parse import parse
from slippi.parse import ParseEvent
from slippi import Game

import os
import glob 
import sys

# get all of the files
path = "dataset/Slippi_Public_Dataset_v3"
all_files = glob.glob(os.path.join(path, "*"))

# find files that are online
def getNetplays(start_index, end_index, all_files):
    netplay_list = []

    for i in range(start_index, end_index):

        file = all_files[i]
        print(i, file=sys.stderr)

        def handle(metadata):
            if metadata.platform.name == 'DOLPHIN':
                netplay_list.append(file)

        handlers = {ParseEvent.METADATA: handle}

        try:
            parse(file, handlers)
        except:
            continue

    return netplay_list

print(getNetplays(0, len(all_files), all_files))
