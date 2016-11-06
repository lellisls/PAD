MPIRUN_FLAGS =-np 4
# FLAGS = -np <processos> -hostfile <arq.marquinas.txt> --prefix /opt/mpi2

#CXX=/usr/lib64/openmpi/bin/mpicc
#MPIRUN=/usr/lib64/openmpi/bin/mpirun
CXX=/opt/mpi2/bin/mpicc
MPIRUN=/opt/mpi2/bin/mpirun
all: maxmin pi_mpi2 hello-mpi

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex02-a: stats
	${CXX} -Iinc -o obj/$@.o src/${@}.c -o bin/$@ -lm 
	${MPIRUN} ${MPIRUN_FLAGS} --prefix /opt/mpi2/ bin/$@

clean:
	rm -rf bin/* obj/*
