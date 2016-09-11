import numpy as np
import sys

switcher = {
    0: "IJK", 1: "IKJ", 2: "JIK",
    3: "JKI", 4: "KIJ", 5: "KJI"
}

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=',', dtype='float64')
    chunks = np.split(data, len(data)/5)
    print "Size, Mode, Time, L2_DCM, MFLOPS, CPI"
    for chunk in chunks:
        row = list(np.mean(chunk, axis=0))
        row[1] = switcher[int(row[1])]
        print "%d, %s, %.1f, %.1f, %.3f, %.2f" % tuple(row)
