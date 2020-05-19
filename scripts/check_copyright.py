from pathlib import Path
files = list(Path("../poprithms").rglob("*.[ch][p][p]"))
bad_files = []
for fl in files:
    with open(fl) as f:
        first_line = f.readline()
        if "Graphcore" not in first_line or "Copyright" not in first_line:
            bad_files.append(str(fl))

#1 : fail, and 0 : succeed
if (not bad_files):
    print("0")

else:
    print("1")
    print("file(s) with missing notices:")
    for x in bad_files:
        print(x)
