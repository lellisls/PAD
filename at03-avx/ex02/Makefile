all: least-squares

CFLAGS=-Iinc obj/* -lpapi -O3 -march=native -fopenmp -ffast-math -ftree-vectorize -fopt-info-vec-all
GREP=2>&1 | grep -e "LOOP VECTORIZED" -e "error" || true

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o -lpapi -O3

least-squares: stats
	gcc src/$@.c -o bin/$@ $(CFLAGS) $(GREP)

least-squares-avx: stats
	gcc src/$@.c -o bin/$@ $(CFLAGS) $(GREP)

# least-squares-avx2: stats
	# gcc src/$@.c -o bin/$@ $(CFLAGS) $(GREP)


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
