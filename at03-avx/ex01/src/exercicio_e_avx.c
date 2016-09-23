#include "stats.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h> /* where intrinsics are defined */
#include <xmmintrin.h>


#ifndef N
#define N 50000000
#endif

int main( ) {
  int i;
  float *z, s = 0.f;
  __m256 data, acc;

  z = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    z[ i ] = ( float ) rand( ) / 100000000.0;
  }
  inicializacao( );
  acc = _mm256_xor_ps( acc, acc );
  for( i = 0; i < N - 8; i += 8 ) {
    data = _mm256_load_ps( z + i );
    acc = _mm256_add_ps( acc, data );
  }
  for( ; i < N; i++ ) {
    s += z[ i ];
  }
  for( i = 0; i < 8; i++ ) {
    s += acc[ i ];
  }
  avaliacao( "Exercicio E - AVX", N );
  PRINT( printf( "%f\n", s ); )

  free( z );
  /* Adaptação para que as operações sobre 's' serem detectadas. */
  return( s * 0 );
}
