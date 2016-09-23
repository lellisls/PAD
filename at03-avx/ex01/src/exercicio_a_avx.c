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
  int i, nvecavx;
  __m256 v1, v2, v3, ones;
  float *x, *y, *z, *a;
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

  nvecavx = N - ( N % 8 );
  ones = _mm256_set1_ps(1.0);
  for( i = 1; i < 8; i++ ) {
    x[ i ] = y[ i ] + z[ i ];
    a[ i ] = x[ i - 1 ] + 1.0;
  }
  for( i = 8; i < N - 8; i += 8 ) {
    // printf("%d\n", i);
    v1 = _mm256_load_ps( y + i );
    v2 = _mm256_load_ps( z + i );
    v3 = _mm256_add_ps( v1, v2 );
    _mm256_store_ps( x + i, v3 );
    // x[ i ] = y[ i ] + z[ i ];

    v1 = _mm256_load_ps( x + i -1 );
    v2 = _mm256_add_ps( v1, ones );
    _mm256_store_ps( a + i, v2 );
    // a[ i ] = x[ i - 1 ] + 1.0;
  }
  for( ; i < N; i++ ) {
    x[ i ] = y[ i ] + z[ i ];
    a[ i ] = x[ i - 1 ] + 1.0;
  }
  avaliacao( "Exercicio A - AVX", N );
  for( i = N - 10; i < N; i++ ) {
    PRINT( printf( "%.2f ", a[ i ] ); );
  }
  PRINT( printf( "\n" ); );

  _mm_free( x );
  _mm_free( y );
  _mm_free( z );
  _mm_free( a );

  return( 0 );
}
