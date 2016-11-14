CFLAGS+= -lm -O3
CCOPTS= -Xcompiler " -fopenmp" -arch=compute_30 -code=sm_30
all: ftcs som_mat ex01 ex02 ex03 ex04

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex01: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	time bin/$@

som_mat: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	time bin/$@

ftcs: stats
	gcc -fopenmp src/${@}.c -o bin/$@ $(CFLAGS)
	time bin/$@

clean:
	rm -rf bin/* obj/*
