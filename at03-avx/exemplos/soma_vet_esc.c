#include <stdio.h>

void sumvetesc(float *a, float *b, float *c, long nvec) {
  long i;

  printf("sumvetesc\n");

  for(i=0; i<nvec; i++)
    c[i] = a[i] + b[i];
}
