import numpy as np
import sys

switcher = {1: "Sem Fusao", 2: "Com fusao"}

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=',', dtype='float64')
    chunks = np.split(data, len(data)/5)
    out = open(sys.argv[2], 'w')
    out.write("Size, Mode, Time, L2_DCM, MFLOPS, CPI\n")
    for chunk in chunks:
        row = list(np.mean(chunk, axis=0))
        row[1] = switcher[int(row[1])]
        out.write("%d, %s, %.1f, %.1f, %.3f, %.2f\n" % tuple(row))
