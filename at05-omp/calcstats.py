import numpy as np
import sys

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=';', dtype='float64')
    mychunks = np.split(data, len(data) / 5)
    out = open(sys.argv[2], 'w')
    cols = ["col{}".format(col) for col in xrange(len(data[0]))]
    out.write(";".join(cols) + "\n")
    # data = []
    for chunk in mychunks:
        row = [str(val) for val in np.mean(chunk, axis=0)]
        out.write(";".join(row) + "\n")
    out.close()
