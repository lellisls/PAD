
# FLAGS = -np <processos> -hostfile <arq.marquinas.txt> --prefix /opt/mpi2

CXX=/usr/lib64/openmpi/bin/mpicc
MPIRUN=/usr/lib64/openmpi/bin/mpirun
MPIRUN_FLAGS =-np 4
MPIRUN_PREFIX=
monkey ?= 0
ifneq ($(monkey), 0)
	CXX=/opt/mpi2/bin/mpicc
	MPIRUN=/opt/mpi2/bin/mpirun
	MPIRUN_FLAGS =-np 14
	MPIRUN_PREFIX = --prefix /opt/mpi2/
endif

all: maxmin pi_mpi2 hello-mpi

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex02-a: stats
	${CXX} -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm
	${MPIRUN} ${MPIRUN_FLAGS} bin/$@ ${MPIRUN_PREFIX}

clean:
	rm -rf bin/* obj/*
