#include "stats.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 50000000
#endif

int main( ) {
  int i;
  float *x, *y, *z, *a, t;
  x = ( float* ) malloc( sizeof( float ) * N );
  y = ( float* ) malloc( sizeof( float ) * N );
  z = ( float* ) malloc( sizeof( float ) * N );
  a = ( float* ) malloc( sizeof( float ) * N );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    x[ i ] = ( float ) rand( ) / 10000000.0;
    y[ i ] = ( float ) rand( ) / 10000000.0;
    z[ i ] = ( float ) rand( ) / 10000000.0;
    a[ i ] = ( float ) rand( ) / 10000000.0;
  }
  inicializacao( );
  for( i = 0; i < N; i++ ) {
    t = y[ i ] + z[ i ];
    a[ i ] = t + 1.0 / t;
  }
  avaliacao( "Exercicio D - Padrão", N );
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
