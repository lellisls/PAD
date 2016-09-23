#include "stats.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 50000000
#endif

int main( ) {
  int i;
  float *a, *b;
  a = ( float* ) malloc( sizeof( float ) * N );
  b = ( float* ) malloc( sizeof( float ) * N );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    a[ i ] = ( float ) rand( ) / 10000000.0;
    b[ i ] = ( float ) rand( ) / 10000000.0;
  }
  inicializacao( );
  for( i = 1; i < N; i++ ) {
    a[ i ] = a[ i - 1 ] + b[ i ];
  }
  avaliacao( "Exercicio F - Padrão", N );
  for( i = N - 10; i < N; i++ ) {
    PRINT( printf( "%.2f ", a[ i ] ); );
  }
  PRINT( printf( "\n" ); );

  free( a );
  free( b );

  return( 0 );
}
