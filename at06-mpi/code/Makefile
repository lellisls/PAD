MPIRUN_FLAGS = -np 4
# FLAGS = -np <processos> -hostfile <arq.marquinas.txt> --prefix /opt/mpi2

CXX=/usr/lib64/openmpi/bin/mpicc
MPIRUN=/usr/lib64/openmpi/bin/mpirun

all: maxmin pi_mpi2 hello-mpi

ex02-a:
	${CXX} -o bin/$@ -lm ${@}.c
	${MPIRUN} ${MPIRUN_FLAGS} bin/$@
