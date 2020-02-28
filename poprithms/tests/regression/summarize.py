import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
import numpy as np

filly = open("path-to-logging-file-gerenated-for-regression.txt")
lines = filly.readlines()

records = {}
description = ""
settingsString = ""
settings = []
for l in lines:
    if "description" in l:
        if (description):
            if (description not in records.keys()):
                records[description] = {}
            if settingsString not in records[description]:
                records[description][settingsString] = []
            records[description][settingsString].append({
                "timeInit": timeInit,
                "timeAnneal": timeAnneal,
                "nOpsBefore": nOpsBefore,
                "nOpsAfter": nOpsAfter
            })
        description = l.split("=")[1].strip()
        settings = []
        settingsString = ""

    elif "timeInitialize" in l:
        timeInit = float(l.split("=")[1].split()[0].strip())
    elif "timeAnneal" in l:
        timeAnneal = float(l.split("=")[1].split()[0].strip())
    elif "nOpsBefore" in l:
        nOpsBefore = int(l.split("=")[1])
    elif "nOpsAfter" in l:
        nOpsAfter = int(l.split("=")[1])
    else:
        print(l)
        settings.append(l)
        settingsString += l

mpl.figure(10, figsize=(8, 9))
mpl.clf()
nPlots = len(records.keys())
gs1 = gridspec.GridSpec(nPlots // 2, 2)
for i, k in enumerate(records.keys()):
    row = int(i // 2)
    col = int(i % 2)
    print(nPlots, i, row, col)
    mpl.subplot(gs1[row:(row + 1), col:(col + 1)])
    mpl.title(k)
    mpl.ylabel("time [s]")
    if i is len(records.keys()) - 1:
        mpl.xlabel("number of Ops")

    print("\n\n")
    print(k)
    for summary in records[k].keys():
        print(summary)
        rs = records[k][summary]

        ax = mpl.gca()

        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=2)

        label = summary.replace('\n', ' ').replace("logging=0 ", "").replace(
            "tieBreaker=", "").replace("pHigherFallRate", "pH").replace(
                "pClimb", "pC").replace("pStayPut", "pS")

        mpl.plot([x["nOpsBefore"] for x in rs], [x["timeAnneal"] for x in rs],
                 linestyle=":",
                 marker="o",
                 label=label)
    mpl.legend()

mpl.show()
