import numpy as np
import sys
import csv

def chunks(l, n):
    n = max(1, n)
    data = []
    for i in xrange(0, len(l), n):
        aux = [cnk[1:] for cnk in l[i:i + n] ]
        data.append( [l[i:i + n][0][0]]+ list(np.mean(aux, axis=0)))
    return data

with open(sys.argv[1], 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    data = []
    out = open(sys.argv[2], 'w')
    out.write("Exemplo, Time, L2_DCM, MFLOPS, CPE\n")
    for row in spamreader:
        newrow = [row[0]] + [float(val) for val in row[1:]]
        data.append(newrow)
    for chunk in chunks(data, 5):
        out.write("%s, %.0f, %.0f, %.4f, %.2f\n" % tuple(chunk))
    out.close()
# with open(sys.argv[1], 'rb') as csvfile:
#     data = np.loadtxt(csvfile, delimiter=',', dtype='float64')
#     mychunks = np.split(data, len(data) / 5)
#     out = open(sys.argv[2], 'w')
#     out.write("Exemplo, Time, L2_DCM, MFLOPS, CPI\n")
#     data = []
#     for chunk in mychunks:
#         row = list(np.mean(chunk, axis=0))
#         data.append(row)
#         out.write("%s, %.1f, %.1f, %.3f, %.2f\n" % tuple(row))
#     out.close()
    # out2 = open(sys.argv[3], 'w')
    # data = sorted(data, key=lambda tup: tup[1])
    # mychunks = chunks(data, 3)
    # for chunk in mychunks:
    #     for row in chunk:
    #         out2.write("%d %.1f\n" % (row[0], row[2]))
    #     out2.write("\n\n")
