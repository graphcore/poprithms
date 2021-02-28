# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
hppfiles = list(Path("../poprithms").rglob("*.[h][p][p]"))
bad_files = []

for fl in hppfiles:
    with open(fl) as f:
        lines = []
        line = f.readline()
        while ("#include" not in line):
            lines.append(line)
            line = f.readline()
        if (lines and lines[-1].strip() != ""):
            bad_files.append(str(fl))

#1 : fail, and 0 : succeed
if (not bad_files):
    print("0")

else:
    print("1")
    print("header file(s) with incorrect header guard spacing are:;")
    for x in bad_files:
        print(x)
