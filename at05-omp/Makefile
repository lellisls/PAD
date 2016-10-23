debug ?= 0
quiet ?= 0
info-vec-all ?= 0
gprof ?= 0

CFLAGS=-Iinc -lpapi -march=native -fopenmp -ffast-math -ftree-vectorize
GREP=

ifneq ($(debug), 0)
	CFLAGS+= -g
else
	CFLAGS+= -O3
endif

ifneq ($(info-vec-all), 0)
	CFLAGS+= -fopt-info-vec-all
	GREP=2>&1 | grep -e "LOOP VECTORIZED" -e "error" || true
endif

ifneq ($(quiet), 0)
	CFLAGS+= -DQUIET
endif

ifneq ($(gprof), 0)
	CFLAGS+= -pg
endif

all: ex01

stats: clean
	gcc-6 -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS) $(GREP)

ex01: stats
	gcc-6 src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP)

ex02: stats
	gcc-6 src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP)

ex03: stats
	gcc-6 src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP)

ex04: stats
	gcc-6 src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP) -DSCHEDULE=${schedule} -DSCHEDULE_TXT=\"${schedule}\"

ex05: stats
	gcc-6 src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP)

clean:
	rm -rf bin/* obj/*
