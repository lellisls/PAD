import numpy as np
import sys

with open(sys.argv[1], 'rb') as csvfile:
    data = np.loadtxt(csvfile, delimiter=',', dtype='float64') 
    chunks = np.split(data, len(data)/5)
    for chunk in chunks:
        row = np.mean(chunk, axis=0)
        if len(row) == 6:
            print "%d, %d, %.1f, %.1f, %.3f, %.2f" % tuple(row)
        else:
            print "%d, %.1f, %.1f, %.3f, %.2f" % tuple(row)




