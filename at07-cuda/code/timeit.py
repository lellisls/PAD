import os
import sys
import time

print sys.argv[1]

start_time = time.time()
times = 3.0

for i in xrange(int(times)):
    os.system(sys.argv[1] + " >> log.txt")
elapsed_time = time.time() - start_time

print elapsed_time/times
