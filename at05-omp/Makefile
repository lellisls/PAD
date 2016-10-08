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

all: life life2

stats: clean
	gcc -c src/$@.c -Iinc -o obj/$@.o $(CFLAGS)

life: stats
	gcc src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP)

life2: stats
	gcc src/$@.c obj/* -o bin/$@ $(CFLAGS) $(GREP)

test: life life2
	bin/life 1000
	bin/life2 1000

clean:
	rm -rf bin/* obj/*
