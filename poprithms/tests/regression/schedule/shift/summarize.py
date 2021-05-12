# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import os


def run(logsDir, plotsDir="."):
    """
    logsDir : 
     -- where to read all log files from (.txt and .log extensions). 
        This is the data which will be plotted.
    plotsDir : 
     -- where to write pdf figures to.
    """

    lines = []
    for fn in [
            os.path.join(logsDir, x) for x in os.listdir(logsDir)
            if ".txt" in x or ".log" in x
    ]:
        filly = open(fn, "r")
        lines += filly.readlines()

    print("In run with ", len(lines), " lines")

    records = {}
    description = ""
    settingsString = ""
    for l in lines:
        if "description" in l:
            if (description):
                if (description not in records.keys()):
                    records[description] = {}
                if settingsString not in records[description]:
                    records[description][settingsString] = []
                records[description][settingsString].append({
                    "timeInit":
                    timeInit,
                    "timeShift":
                    timeShift,
                    "nOpsBefore":
                    nOpsBefore,
                    "nOpsAfter":
                    nOpsAfter
                })
            description = l.split("=")[1].strip()
            settingsString = ""

        elif "timeInitialize" in l:
            timeInit = float(l.split("=")[1].split()[0].strip())
        elif "timeShift" in l:
            timeShift = float(l.split("=")[1].split()[0].strip())
        elif "nOpsBefore" in l:
            nOpsBefore = int(l.split("=")[1])
        elif "nOpsAfter" in l:
            nOpsAfter = int(l.split("=")[1])
        else:
            #shorten the string for cleaner figure legend:
            if "logTime=" in l:
                l = l.split("logTime=")[1].split("at")[0]
            settingsString += l

    nPlots = len(records.keys())
    for i, k in enumerate(records.keys()):
        gs1 = gridspec.GridSpec(1, 1)
        mpl.subplot(gs1[0:1, 0:1])
        mpl.title(k)
        mpl.ylabel("time [s]")
        mpl.xlabel("number of Ops")
        for summary in records[k].keys():
            rs = records[k][summary]
            ax = mpl.gca()
            ax.set_xscale('log', basex=2)
            ax.set_yscale('log', basey=2)

            label = summary.replace('\n', ' ').replace("logging=0 ",
                                                       "").replace(
                                                           "tieBreaker=", "")

            mpl.plot([x["nOpsBefore"] for x in rs],
                     [x["timeShift"] + x["timeInit"] for x in rs],
                     linestyle=":",
                     marker="o",
                     label=label)
        mpl.legend(loc="lower right")

        plotfilename = os.path.join(plotsDir, "%s.pdf" % (k, ))
        print("Saving figure at ", plotfilename)
        mpl.savefig(plotfilename)


if __name__ == "__main__":
    # expected use is something like
    #  >>  python3 summarize.py logs/ plots/
    if (len(sys.argv) != 3):
        raise RuntimeError(
            "Expected 2 arguments: (0) where the log files are and (1) where to store pdf plots"
        )

    logsDir = sys.argv[1]
    plotsDir = sys.argv[2]
    run(logsDir, plotsDir)
