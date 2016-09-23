#include <stdio.h>
#include <emmintrin.h> /* where intrinsics are defined */
#include <xmmintrin.h>

#define MATLEN 100000000

int main() {

  float *a, *b, *c; //, lixo[4];
  long i;
  __m128 v1, v2, v3;

  a = (float *) _mm_malloc(MATLEN*sizeof(float),16);
  b = (float *) _mm_malloc(MATLEN*sizeof(float),16);
  c = (float *) _mm_malloc(MATLEN*sizeof(float),16);

  a[0] = 1.2; b[0] = 4.5;
  a[1] = 0.5; b[1] = 0.7;
  a[2] = 1.5; b[2] = 1.7;

  printf("Alocacao efetuada\n");

  // sumvetesc(a, b, c, MATLEN);
  sumvetsse(a, b, c, MATLEN);

  printf("%f\n%f\n%f\n", c[0], c[1], c[2]);

  printf("Fim\n");
  return(0);
}
