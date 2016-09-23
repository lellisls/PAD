#include "stats.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 50000000
#endif

int main( ) {
  int i;
  float *x, *z, *a;
  x = ( float* ) malloc( sizeof( float ) * N );
  z = ( float* ) malloc( sizeof( float ) * N );
  a = ( float* ) malloc( sizeof( float ) * N );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    x[ i ] = ( float ) rand( ) / 10000000.0;
    z[ i ] = ( float ) rand( ) / 10000000.0;
    a[ i ] = ( float ) rand( ) / 10000000.0;
  }
  inicializacao( );
  for( i = 0; i < N - 1; i++ ) {
    x[ i ] = a[ i ] + z[ i ];
    a[ i ] = x[ i + 1 ] + 1.0;
  }
  avaliacao( "Exercicio C - Padrão", N );
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
