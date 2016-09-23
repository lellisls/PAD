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
  float *x, *y, *z, *a, t;
  __m256 v1, v2, v3, ones;
  x = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  y = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  z = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  a = ( float* ) _mm_malloc( sizeof( float ) * N, 32 );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    x[ i ] = ( float ) rand( ) / 10000000.0;
    y[ i ] = ( float ) rand( ) / 10000000.0;
    z[ i ] = ( float ) rand( ) / 10000000.0;
    a[ i ] = ( float ) rand( ) / 10000000.0;
  }
  inicializacao( );

  ones = _mm256_set1_ps( 1.0 );
  for( i = 0; i < N - 8; i += 8 ) {
    v1 = _mm256_load_ps( y + i );
    v2 = _mm256_load_ps( z + i );
    v1 = _mm256_add_ps( v1, v2 );
    /* t = y[ i ] + z[ i ]; */
    v2 = _mm256_div_ps( ones, v1 );
    v1 = _mm256_add_ps( v1, v2 );
    _mm256_store_ps( a + i, v1 );
    /* a[ i ] = t + 1.0 / t; */
  }
  for( ; i < N; i++ ) {
    t = y[ i ] + z[ i ];
    a[ i ] = t + 1.0 / t;
  }
  avaliacao( "Exercicio D - AVX", N );
  for( i = N - 10; i < N; i++ ) {
    PRINT( printf( "%.2f ", a[ i ] ); );
  }
  PRINT( printf( "\n" ); );
  free( x );
  free( y );
  free( z );
  free( a );

  return( 0 );
}
