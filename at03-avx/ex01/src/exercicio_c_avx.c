#include "stats.h"
#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h> /* where intrinsics are defined */
#include <xmmintrin.h>


#ifndef N
#define N 50000000
#endif

int main( ) {
  int i;
  float *x, *z, *a;
  __m256 v1, v2, v3, ones;

  x = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  z = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  a = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    x[ i ] = ( float ) rand( ) / 10000000.0;
    z[ i ] = ( float ) rand( ) / 10000000.0;
    a[ i ] = ( float ) rand( ) / 10000000.0;
  }
  inicializacao( );

  ones = _mm256_set1_ps( 1.0 );
  for( i = 0; i < N - 9; i += 8 ) {
    v1 = _mm256_load_ps( a + i );
    v2 = _mm256_load_ps( z + i );
    v3 = _mm256_add_ps( v1, v2 ); //aux

    v1 = _mm256_load_ps( x + i + 1);
    v1 = _mm256_add_ps( v1, ones);
    _mm256_store_ps( a + i, v1 );
    _mm256_store_ps( x + i, v3 );
  }
  for( ; i < N - 1; i++ ) {
    x[ i ] = a[ i ] + z[ i ];
    a[ i ] = x[ i + 1 ] + 1.0;
  }
  avaliacao( "Exercicio C - AVX", N );
  for( i = N - 10; i < N; i++ ) {
    PRINT( printf( "%.2f ", a[ i ] ); );
  }
  PRINT( printf( "\n" ); );
  for( i = N - 10; i < N; i++ ) {
    PRINT( printf( "%.2f ", x[ i ] ); );
  }
  PRINT( printf( "\n" ); );
  free( x );
  free( z );
  free( a );

  return( 0 );
}
