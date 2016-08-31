#include <stdio.h>

#define MATLEN 500
#define block 10

int main( ) {

  int i, j, k;
  int ii, jj, kk;
  double **a, **b, **c;

  a = ( double** ) malloc( MATLEN * sizeof( double* ) );
  b = ( double** ) malloc( MATLEN * sizeof( double* ) );
  c = ( double** ) malloc( MATLEN * sizeof( double* ) );
  for( i = 0; i < MATLEN; i++ ) {
    a[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
    b[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
    c[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
  }
/*   for (i=0; i<MATLEN; i++) { */
/*     for (j=0; j<MATLEN; j++) { */
/*       for (k=0; k<MATLEN; k++) { */
/*      c[i][j] = c[i][j] + a[i][k]*b[k][j]; */
/*       } */
/*     } */
/*   } */
  for( ii = 0; ii < MATLEN; ii += block ) {
    for( jj = 0; jj < MATLEN; jj += block ) {
      for( kk = 0; kk < MATLEN; kk += block ) {
        for( i = ii; i < ii + block; i++ ) {
          for( j = jj; j < jj + block; j++ ) {
            for( k = kk; k < kk + block; k++ ) {
              c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
            }
          }
        }
      }
    }
  }
  printf( "Fim\n" );
  return( 0 );
}
