CXX=nvcc
CFLAGS+= -lm -O3

all: ex01

stats: clean
	# gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex01: stats
	${CXX} -Iinc src/${@}.cu -o bin/$@ $(CFLAGS)
	bin/$@

clean:
	rm -rf bin/* obj/*
