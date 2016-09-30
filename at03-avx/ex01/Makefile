all: tests

debug ?= 0
quiet ?= 0
info-vec-all ?= 0
gprof ?= 0


CFLAGS=-Iinc -lpapi -O3 -march=native -fopenmp -ffast-math -ftree-vectorize
GREP=

ifneq ($(info-vec-all), 0)
	CFLAGS+= -fopt-info-vec-all
	GREP=2>&1 | grep -e "LOOP VECTORIZED" -e "error" || true
endif

ifneq ($(quiet), 0)
	CFLAGS+= -DQUIET
endif

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

exercicio_a: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_a_avx: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_b: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_b_mod: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_b_avx: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_c: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_c_mod: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_c_avx: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_d: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_d_avx: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_e: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_e_avx: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicio_f: stats
	gcc src/$@.c -o bin/$@ obj/* $(CFLAGS) $(GREP)

exercicios: exercicios_normal exercicios_mod exercicios_avx
exercicios_normal: exercicio_a exercicio_b exercicio_c exercicio_d exercicio_e exercicio_f
exercicios_mod: exercicio_b_mod exercicio_c_mod
exercicios_avx: exercicio_a_avx exercicio_b_avx exercicio_c_avx exercicio_d_avx exercicio_e_avx

tests: test_a test_b test_c test_d test_e test_f

test_a: exercicio_a exercicio_a_avx
	bin/exercicio_a
	bin/exercicio_a_avx

test_b: exercicio_b exercicio_b_mod exercicio_b_avx
	bin/exercicio_b
	bin/exercicio_b_mod
	bin/exercicio_b_avx

test_c: exercicio_c exercicio_c_mod exercicio_c_avx
	bin/exercicio_c
	bin/exercicio_c_mod
	bin/exercicio_c_avx

test_d: exercicio_d exercicio_d_avx
	bin/exercicio_d
	bin/exercicio_d_avx

test_e: exercicio_e exercicio_e_avx
	bin/exercicio_e
	bin/exercicio_e_avx

test_f: exercicio_f
	bin/exercicio_f

clean:
	rm -rf bin/* obj/*
