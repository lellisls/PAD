CFLAGS+= -lm -O3
CCOPTS= -std=c++11 -arch=compute_30 -code=sm_30 -I /usr/include/cuda/ -Xcompiler " -fopenmp" -D__CORRECT_ISO_CPP11_MATH_H_PROTO
all: ftcs som_mat ex01 ex02 ex03 ex04

ex01: clean
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@
	gcc -fopenmp -Iinc src/${@}.c -o bin/$@ $(CFLAGS)
	bin/$@

ex01-b: clean
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

ex03: clean
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@
	gcc -fopenmp -Iinc src/${@}.c -o bin/$@ $(CFLAGS)
	bin/$@

ex04: clean
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

ex05: clean
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@
	nvcc $(CCOPTS) -Iinc src/${@}-b.cu -o bin/$@ $(CFLAGS)
	bin/$@
	gcc -fopenmp -Iinc src/${@}.c -o bin/$@ $(CFLAGS)
	bin/$@

som_mat: clean
	nvcc $(CCOPTS) -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

clean:
	rm -rf bin/* obj/*
