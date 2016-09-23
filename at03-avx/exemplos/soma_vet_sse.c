#include <emmintrin.h> /* where intrinsics are defined */
#include <xmmintrin.h>
#include <stdio.h>

void sumvetsse(float *a, float *b, float *c, long int nvec) {
  long int i, nvecsse;
  __m128 v1, v2;

  printf("sumvetsse\n");

  nvecsse = nvec - (nvec%4);
  for(i=0; i<nvec; i+=4) {
    v1 = _mm_load_ps(a+i);
    v2 = _mm_load_ps(b+i);
    v2 = _mm_add_ps(v1, v2);
    _mm_store_ps(c+i, v2);
  }
  for(i=nvecsse; i<nvec; i++)
    c[i] = a[i] + b[i];
}
