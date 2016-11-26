CFLAGS+= -lm -O3
CCOPTS= -std=c++11 -arch=compute_30 -code=sm_30 -I /usr/include/cuda/ -Xcompiler " -fopenmp" -D__CORRECT_ISO_CPP11_MATH_H_PROTO
all: ftcs som_mat ex01 ex02 ex03 ex04

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex01: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@
	gcc -fopenmp -Iinc src/${@}.c -o bin/$@ $(CFLAGS)
	bin/$@

ex01-b: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

ex03: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@
	gcc -fopenmp -Iinc src/${@}.c -o bin/$@ $(CFLAGS)
	bin/$@

ex04: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

ex05: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@
	gcc -fopenmp -Iinc src/${@}.c -o bin/$@ $(CFLAGS)
	bin/$@

som_mat: stats
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

clean:
	rm -rf bin/* obj/*
