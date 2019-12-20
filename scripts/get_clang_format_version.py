"""
A Python program to print the major version of clang-format
"""

import sys
import subprocess
out = subprocess.run(["clang-format", "--version"], stdout=subprocess.PIPE)

#the string output of running clang-format --version
stdoutput = out.stdout.decode('utf-8')

#Assuming there is a "version a.b.c" in the output,
#this should be something like 8.0.0
version_string = stdoutput.split("version")[-1].strip().split()[0]

version_major = version_string.split(".")[0]
print(int(version_major))
