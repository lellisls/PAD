import numpy as np
import sys

switcher = {
    1: "abc[ ? ][3]", 2: "abc[3][ ? ]", 3: "struct",
}

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=',', dtype='float64')
    chunks = np.split(data, len(data)/5)
    print "Size, Mode, Time, L2_DCM, MFLOPS, CPI"
    for chunk in chunks:
        row = list(np.mean(chunk, axis=0))
        row[1] = switcher[int(row[1])]
        print "%d, %s, %.1f, %.1f, %.3f, %.2f" % tuple(row)
