#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import argparse

################################################################################
# usage: plot.py [-h] -e EXE -f FRONT_FILE -d TEST_DATASET [-t TITLE]
#                [-x XLABEL] [-y YLABEL] [-fbl FIRST_BAR_LABEL]
#                [-sbl SECOND_BAR_LABEL] [-out OUT_FILE] [-dup] [-min MIN]
#                [-max MAX]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -e EXE, --exe EXE     Executable filename
#   -f FRONT_FILE, --front-file FRONT_FILE
#                         Pareto-front file
#   -d TEST_DATASET, --test-dataset TEST_DATASET
#                         Test dataset
#   -t TITLE, --title TITLE
#                         Title of the figure [default=none]
#   -x XLABEL, --xlabel XLABEL
#                         Label of the x-axis [default='Complexity (nodes)']
#   -y YLABEL, --ylabel YLABEL
#                         Label of the y-axis [default='Error']
#   -fbl FIRST_BAR_LABEL, --first-bar-label FIRST_BAR_LABEL
#                         Label of the first bar [default='Training set']
#   -sbl SECOND_BAR_LABEL, --second-bar-label SECOND_BAR_LABEL
#                         Label of the second bar [default='Test set']
#   -out OUT_FILE, --out-file OUT_FILE
#                         Figure output filename [default=plot.pdf]
#   -dup, --plot-duplicate
#                         Plot duplicate entries instead of taking their mean
#                         [default=false]
#   -min MIN, --min MIN   Ignore sizes less than the given minimum size
#                         [default=0]
#   -max MAX, --max MAX   Ignore sizes greater than the given maximum size
#                         [default=inf]
###
# Example:
#
#    python script/plot.py -e build/speed -f build/speed.front -d build/test.csv
################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exe", required=True, help="Executable filename")
parser.add_argument("-f", "--front-file", required=True, help="Pareto-front file")
parser.add_argument("-d", "--test-dataset", required=True, help="Test dataset")
parser.add_argument("-t", "--title", default="", help="Title of the figure [default=none]")
parser.add_argument("-x", "--xlabel", default="Complexity (nodes)", help="Label of the x-axis [default='Complexity (nodes)']")
parser.add_argument("-y", "--ylabel", default="Error", help="Label of the y-axis [default='Error']")
parser.add_argument("-fbl", "--first-bar-label", default="Training set", help="Label of the first bar [default='Training set']")
parser.add_argument("-sbl", "--second-bar-label", default="Test set", help="Label of the second bar [default='Test set']")
parser.add_argument("-out", "--out-file", default="plot.pdf", help="Figure output filename [default=plot.pdf]")
parser.add_argument("-dup", "--plot-duplicate", action='store_true', default=False, help="Plot duplicate entries instead of taking their mean [default=false]")
parser.add_argument("-min", "--min", type=int, default=0, help="Ignore sizes less than the given minimum size [default=0]")
parser.add_argument("-max", "--max", type=int, default=sys.maxsize, help="Ignore sizes greater than the given maximum size [default=inf]")
args = parser.parse_args()

try:
   f = open(args.front_file,"r")
except IOError:
   print("Could not open file '" + args.front_file + "'", file=sys.stderr)
   sys.exit(1)
lines = f.readlines()
f.close()

size      = []
tra_error = []
solution  = []

for i in range(len(lines)):
   sz = int(lines[i].split(';')[1])
   if sz >= args.min and sz <= args.max:
      size.append(sz)
      tra_error.append(lines[i].split(';')[2])
      solution.append(lines[i].split(';')[4])
size = np.array(size).astype(int)
tra_error = np.array(tra_error).astype(float)

try:
   f = open(args.test_dataset,"r")
except IOError:
   print("Could not open file '" + args.test_dataset + "'", file=sys.stderr)
   sys.exit(1)
lines = f.readlines()
f.close()


tes_error = np.empty(len(solution)); 
print("# training error -> '%s'" % (args.first_bar_label))
print("# test error     -> '%s'" % (args.second_bar_label))
print("ID    size :: training error :: test error     :: solution (encoded terminals)")
print("--    ----    --------------    --------------    ----------------------------")
for i in range(len(solution)):
    tes_error[i] = subprocess.check_output(["./" + args.exe, "-d", args.test_dataset, "-sol", str(size[i]) + " " + str(solution[i].rstrip())])
    print("[%-3d] %-4s :: %-14s :: %-14s :: %s" % (i+1, size[i], tra_error[i], tes_error[i], str(solution[i].rstrip())))

min_value = sys.float_info.max; max_value = 0.

fig = plt.figure(figsize=(18,4))
ax = fig.add_subplot(111)
if args.title:
   ax.set_title(args.title, fontsize=14, fontweight='bold')
if args.xlabel:
   ax.set_xlabel(args.xlabel, fontweight='bold', fontsize=12)
if args.ylabel:
   ax.set_ylabel(args.ylabel, fontweight='bold', fontsize=12)
width = 0.4

vector = []; marks = []
for i in range(1,max(size[tes_error<1.e+30])+1):
   if args.plot_duplicate:
       if len(tra_error[(tes_error<1.e+30)]) > 0:
           for complexity in tra_error[(tes_error<1.e+30) & (size==i)]:
              marks.append(i)
              vector.append(complexity)
   else:
       if len(tra_error[(tes_error<1.e+30) & (size==i)]) > 0:
           marks.append(i)
           vector.append(np.mean(tra_error[(tes_error<1.e+30) & (size==i)]))
ax.bar(list(range(1,len(marks)+1)), vector, width, color='b', label=args.first_bar_label)

vector = []
for i in range(1,max(size[tes_error<1.e+30])+1):
   if args.plot_duplicate:
       if len(tes_error[(tes_error<1.e+30)]) > 0:
           for complexity in tes_error[(tes_error<1.e+30) & (size==i)]:
              vector.append(complexity)
   else:
       if len(tes_error[(tes_error<1.e+30) & (size==i)]) > 0:
           vector.append(np.mean(tes_error[(tes_error<1.e+30) & (size==i)]))
ax.bar([x + width for x in range(1,len(marks)+1)], vector, width, color='r', label=args.second_bar_label)

if min(tra_error) < min_value: min_value = min(tra_error)
if max(tra_error) > max_value: max_value = max(tra_error)
if min(tes_error) < min_value: min_value = min(tes_error)
if max(tes_error[tes_error<1.e+30]) > max_value: max_value = max(tes_error[tes_error<1.e+30])

ax.set_ylim(min_value-0.03*min_value,max_value+0.03*max_value)
ax.set_xlim(width,len(marks)+4*width)
ax.set_xticks([x + width for x in range(1,len(marks)+1)])  
xtickNames = ax.set_xticklabels(marks)
plt.setp(xtickNames, rotation=45, fontsize=8)

ax.legend(loc='upper right', scatterpoints=1, ncol=1, fontsize=12)
fig.savefig(args.out_file, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', papertype='letter', bbox_inches='tight')

print("\nPlotted to file '%s'" % (args.out_file))
#plt.show()
