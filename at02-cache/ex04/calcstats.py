import numpy as np
import sys

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=',', dtype='float64')
    chunks = np.split(data, len(data)/5)
    out = open(sys.argv[2], 'w')
    out.write("Size, Time, L2_DCM, MFLOPS, CPI\n")
    data = []
    for chunk in chunks:
        row = np.mean(chunk, axis=0)
        data.append(row)
        out.write("%d, %.1f, %.1f, %.3f, %.2f\n" % tuple(row))

    out2 = open(sys.argv[3], 'w')
    for row in data:
        out2.write("%d %.1f\n" % (row[0], row[1]))
