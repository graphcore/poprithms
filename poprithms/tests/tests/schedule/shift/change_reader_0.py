# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import sys
import os
from pathlib import Path

# The code being tested in this test file is in the poprithms/notes directory
# of the source tree:
currentSourceDir = Path(os.path.dirname(__file__))
projectDir = currentSourceDir.parent.parent.parent.parent
shiftNotestDir = os.path.join(projectDir, "notes/schedule/shift")
if not os.path.isdir(shiftNotestDir):
    raise RuntimeException("Failed to locate shift notes directory")
sys.path.append(shiftNotestDir)
from shift_util import *

inTestMode = True
makeMarkdown = not inTestMode


def getGenDirs():
    """These are the files names set in change_writer_0"""
    return [
        "gridLog", "treeLog", "recomputeLog", "adversaryLog", "bifurcatingLog"
    ]


def applyToDir(bdRead):
    """
    Test loading a few files which were written by change_writer_0
    If makeMarkdown is True, then diagrams are generated
    """

    print("Will read the data written by change_writer_0 from :", bdRead)

    initialScheduleFn = os.path.join(bdRead, "initialSchedule.txt")
    print("Loading schedule from", initialScheduleFn)
    initialSchedule = getSchedule(initialScheduleFn)

    finalScheduleFn = os.path.join(bdRead, "finalSchedule.txt")
    print("Loading schedule from", finalScheduleFn)
    finalSchedule = getSchedule(finalScheduleFn)

    shiftsFn = os.path.join(bdRead, "shifts.txt")
    print("Loading shifts from", shiftsFn)
    shifts = getShifts(shiftsFn)

    livenessesFn = os.path.join(bdRead, "livenessProfiles.txt")
    print("Loading livenesses from", livenessesFn)
    livenesses = getLivenesses(livenessesFn)

    assert len(livenesses) == len(shifts) + 1
    assert len(initialSchedule) == len(livenesses[0])
    assert len(finalSchedule) == len(livenesses[0])

    #if 'import json' fails, these graphs are none
    graphFromUserFn = os.path.join(bdRead, "graphFromUser.json")
    print("Loading graph from", graphFromUserFn)
    graphFromUser = getGraph(graphFromUserFn)

    graphPreShiftingFn = os.path.join(bdRead, "graphPreShifting.json")
    print("Loading graph from", graphPreShiftingFn)
    graphPreShifting = getGraph(graphPreShiftingFn)

    poly = getLivenessesPoly([3, 5])
    assert poly == [[0, 0], [0, 3], [1, 3], [1, 5], [2, 5], [2, 0]]

    # Used for generating final diagrams
    if (makeMarkdown):
        import visualization
        print(
            "Entering visualization generation stage because makeMarkdown=True"
        )
        writeDir = os.path.join(bdRead, "animation")
        if (not os.path.exists(writeDir)):
            os.mkdir(writeDir)
        visualization.makeMarkdown(graphPreShifting, livenesses, shifts,
                                   initialSchedule, finalSchedule, bdRead,
                                   writeDir)


if inTestMode:
    # This is the file written to by change_writer_0. It contains the name of the
    # file where the log files were written.
    filly = open("dataWriteDir.txt")
    bdRead = filly.readlines()[0]
    applyToDir(bdRead)

else:
    for genDir in getGenDirs():
        applyToDir(genDir)
