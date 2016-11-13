
# FLAGS = -np <processos> -hostfile <arq.marquinas.txt> --prefix /opt/mpi2

CXX=/usr/lib64/openmpi/bin/mpicc
MPIRUN=/usr/lib64/openmpi/bin/mpirun
MPIRUN_FLAGS =-np 4
MPIRUN_FLAGS_OMP =-np 1 --bind-to none
MPIRUN_PREFIX=
monkey ?= 0
CFLAGS+= -lm -O3
ifneq ($(monkey), 0)
	CXX=/opt/mpi2/bin/mpicc
	MPIRUN=/opt/mpi2/bin/mpirun
	MPIRUN_FLAGS =-np 14 -hostfile maquinas.txt
	MPIRUN_PREFIX = --prefix /opt/mpi2/
	CFLAGS+= -DBOARDSIZE=1000
endif

all: ex02-a ex04

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex02-a: stats
	${CXX} -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS)
	time ${MPIRUN} ${MPIRUN_FLAGS} bin/$@ ${MPIRUN_PREFIX}

ex02-b: stats
	${CXX} -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS)
	time ${MPIRUN} ${MPIRUN_FLAGS} bin/$@ ${MPIRUN_PREFIX}

ex03-a: stats
	${CXX} -fopenmp -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS)
	time ${MPIRUN} ${MPIRUN_FLAGS_OMP} bin/$@ ${MPIRUN_PREFIX}

ex04: stats
	${CXX}  -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm $(CFLAGS)
	# ${MPIRUN} ${MPIRUN_FLAGS} bin/$@ ${MPIRUN_PREFIX}

ex04-test1: ex04
	cat histogram.in | time ${MPIRUN} ${MPIRUN_FLAGS} bin/ex04 ${MPIRUN_PREFIX}

ex04-test2: ex04
	cat large-image.ppm | time ${MPIRUN} -np 1 bin/ex04 ${MPIRUN_PREFIX}

clean:
	rm -rf bin/* obj/*
