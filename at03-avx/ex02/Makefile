all: least-squares least-squares-avx

debug ?= 0
quiet ?= 0
info-vec-all ?= 0
gprof ?= 0


CFLAGS=-Iinc  -I/home-local/aluno/papi/include -L/home-local/aluno/papi/lib -lpapi -O3 -march=native -fopenmp -ffast-math -ftree-vectorize
GREP=2>&1 | grep -e "LOOP VECTORIZED" -e "error" || true


ifneq ($(info-vec-all), 0)
        CFLAGS+= -fopt-info-vec-all
        GREP=2>&1 | grep -e "LOOP VECTORIZED" -e "error" || true
endif

ifneq ($(quiet), 0)
        CFLAGS+= -DQUIET
endif


stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

least-squares: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

least-squares-avx: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

# least-squares-avx2: stats
	# gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)


test1: least-squares least-squares-avx #least-squares-avx2
	bin/least-squares tests/test1.txt
	bin/least-squares-avx tests/test1.txt
	# bin/least-squares-avx2 tests/test1.txt

test2: least-squares least-squares-avx #least-squares-avx2
	bin/least-squares
	bin/least-squares-avx
	# bin/least-squares-avx2

tests: test1 test2

clean:
	rm -rf bin/* obj/*
