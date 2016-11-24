

CXX=mpicc
MPIRUN=mpirun
MPIRUN_FLAGS =-np 6
MPIRUN_FLAGS_OMP =-np 1
MPIRUN_PREFIX=
CFLAGS+= -lm -O3

numthds ?= 1

monkey ?= 0
ifneq ($(monkey), 0)
	CXX=/opt/mpi2/bin/mpicc
	MPIRUN=/opt/mpi2/bin/mpirun
	MPIRUN_FLAGS =-np 14 -hostfile maquinas.txt
	MPIRUN_FLAGS_OMP =-np 3 -hostfile maquinas-omp.txt
	MPIRUN_PREFIX = --prefix /opt/mpi2/
	CFLAGS+= -DBOARDSIZE=2000
endif

all: ex02-a ex02-b ex03 ex04

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex02-a: stats
	${CXX} -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS)
	# bash sendfiles.sh hostfiles/maquinas-all.txt bin/$@ /tmp/$@

ex02-b: stats
	${CXX} -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS)
	# bash sendfiles.sh hostfiles/maquinas-all.txt bin/$@ /tmp/$@

ex03: stats
	${CXX} -fopenmp -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS) -DNUMTHDS=${numthds}
	# bash sendfiles.sh hostfiles/maquinas-all.txt bin/$@ /tmp/$@

ex04: stats
	${CXX} -fopenmp -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS) -DNUMTHDS=${numthds}
	# bash sendfiles.sh hostfiles/maquinas-all.txt bin/$@ /tmp/$@

ex04-test1: ex04
	cat histogram.in | time ${MPIRUN} ${MPIRUN_FLAGS} bin/ex04 ${MPIRUN_PREFIX}

ex04-test2: ex04
	cat large-image.ppm | time ${MPIRUN} -np 1 bin/ex04 ${MPIRUN_PREFIX}

clean:
	rm -rf bin/* obj/*
