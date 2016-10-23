import numpy as np
import sys
import csv


def chunks(l, n):
    n = max(1, n)
    data = []
    for i in xrange(0, len(l), n):
        aux = [cnk[1:] for cnk in l[i:i + n]]
        data.append([l[i:i + n][0][0]] + list(np.mean(aux, axis=0)))
    return data

with open(sys.argv[1], 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    out = open(sys.argv[2], 'w')
    data = []
    for row in spamreader:
        newrow = [row[0]] + [float(val) for val in row[1:]]
        data.append(newrow)
    cols = ["col{}".format(col) for col in xrange(len(data[0]))]
    out.write(";".join(cols) + "\n")
    for chunk in chunks(data, 5):
        row = [str(val) for val in chunk]
        out.write(";".join(row) + "\n")
    out.close()
