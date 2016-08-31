#include <cblas.h>
#include <stdio.h>

/* compilar: gcc fonte.c -o exec -lblas */


int main( void ) {
  int lda = 3;
  float A[] = { 1, 0, -1,
                2, 3, 1 };

  int ldb = 2;
  float B[] = { 5, 1,
                6, -1,
                7, 2 };

  int ldc = 2;
  float C[] = { 0.00, 0.00,
                0.00, 0.00 };

  /* Compute C = A B */

  cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3, 1.0, A, lda, B, ldb, 0.0, C, ldc );

  printf( "[ %g, %g\n", C[ 0 ], C[ 1 ] );
  printf( "  %g, %g ]\n", C[ 2 ], C[ 3 ] );

  return( 0 );
}
