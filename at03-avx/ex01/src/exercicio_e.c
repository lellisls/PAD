#include "stats.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 50000000
#endif

int main( ) {
  int i;
  float *z, s = 0.f;
  z = ( float* ) malloc( sizeof( float ) * N );
  srand( 424242 );
  for( i = 0; i < N; i++ ) {
    z[ i ] = ( float ) rand( ) / 100000000.0;
  }
  inicializacao( );
  for( i = 0; i < N; i++ ) {
    s += z[ i ];
  }
  avaliacao( "Exercicio E - Padrão", N );
  PRINT(printf("%f\n", s);)

  free( z );
  /* Adaptação para que as operações sobre 's' serem detectadas. */
  return( s * 0 );
}
