#1/bin/bash
set -e
aux="--hostfile hostfiles/maquinas-all.txt"

# make ex02-a ex02-b > log.txt
# bash sendfiles.sh hostfiles/maquinas-all.txt bin/ex02-a /tmp/ex02-a
# bash sendfiles.sh hostfiles/maquinas-all.txt bin/ex02-b /tmp/ex02-b

#
# python timeit.py "mpirun -np 1 /tmp/ex02-a";
# python timeit.py "mpirun -np 1 /tmp/ex02-b";
#
# python timeit.py "mpirun -np 4 /tmp/ex02-a"
# python timeit.py "mpirun -np 4 /tmp/ex02-b"
#
# python timeit.py "mpirun -np 2 $aux /tmp/ex02-a"
# python timeit.py "mpirun -np 2 $aux /tmp/ex02-b"
#
# python timeit.py "mpirun -np 4 $aux /tmp/ex02-a"
# python timeit.py "mpirun -np 4 $aux /tmp/ex02-b"
#
# python timeit.py "mpirun -np 8 $aux /tmp/ex02-a"
# python timeit.py "mpirun -np 8 $aux /tmp/ex02-b"
#
# make ex03 numthds=1 >> log.txt
# bash sendfiles.sh hostfiles/maquinas-all.txt bin/ex03 /tmp/ex03
# python timeit.py "mpirun -np 1 /tmp/ex03";
# python timeit.py "mpirun -np 4 /tmp/ex03"
# python timeit.py "mpirun -np 8 $aux /tmp/ex03"
#
# make ex03 numthds=4 >> log.txt
# bash sendfiles.sh hostfiles/maquinas-all.txt bin/ex03 /tmp/ex03
# python timeit.py "mpirun -np 1 /tmp/ex03";
# python timeit.py "mpirun -np 2 $aux /tmp/ex03"

make ex04 numthds=1 >> log.txt
bash sendfiles.sh hostfiles/maquinas-all.txt bin/ex04 /tmp/ex04

python timeit.py "mpirun -np 1 /tmp/ex04";
python timeit.py "mpirun -np 4 /tmp/ex04"
python timeit.py "mpirun -np 8 $aux /tmp/ex04"

make ex04 numthds=4 >> log.txt
bash sendfiles.sh hostfiles/maquinas-all.txt bin/ex04 /tmp/ex04
python timeit.py "mpirun -np 1 /tmp/ex04";
python timeit.py "mpirun -np 2 $aux /tmp/ex04"
