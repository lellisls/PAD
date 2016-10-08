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
    spamreader = csv.reader(csvfile, delimiter=',')
    data = []
    out = open(sys.argv[2], 'w')
    out.write("Exemplo, Tempo, CPE\n")
    for row in spamreader:
        newrow = [row[0]] + [float(val) for val in row[1:]]
        data.append(newrow)
    for chunk in chunks(data, 5):
        out.write("%s, %.4f, %.4f\n" % tuple(chunk))
    out.close()
