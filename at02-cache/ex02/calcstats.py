import numpy as np
import sys


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in xrange(0, len(l), n)]

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=',', dtype='float64')
    mychunks = np.split(data, len(data) / 5)
    out = open(sys.argv[2], 'w')
    out.write("Size, Block, Time, L2_DCM, MFLOPS, CPI\n")
    data = []
    for chunk in mychunks:
        row = np.mean(chunk, axis=0)
        data.append(row)
        out.write("%d, %d, %.1f, %.1f, %.3f, %.2f\n" % tuple(row))

    out2 = open(sys.argv[3], 'w')
    data = sorted(data, key=lambda tup: tup[1])
    mychunks = chunks(data, 3)
    for chunk in mychunks:
        for row in chunk:
            out2.write("%d %.1f\n" % (row[0], row[2]))
        out2.write("\n\n")
