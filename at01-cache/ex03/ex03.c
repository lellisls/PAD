#include <stdio.h>
#include <stdlib.h>

#ifndef MATLEN
#define MATLEN 1024
#endif

#ifndef BLOCK
#define BLOCK 16
#endif

#ifndef QUIET
#define PRINT( exp ) exp
#else
#define PRINT( exp )
#endif

#include "papi.h"


void multiplyBlock( double **a, double **b, double **c, int n ) {
  int i, j, k;
  for( i = 0; i < n; ++i ) {
    for( k = 0; k < n; ++k ) {
      for( j = 0; j < n; ++j ) {
        c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
      }
    }
  }
}

double** add( double **a, double **b, int n ) {
  double **c;
  int i, j;
  c = ( double** ) malloc( n * sizeof( double* ) );
  for( i = 0; i < n; i++ ) {
    c[ i ] = ( double* ) malloc( n * sizeof( double ) );
  }
  for( i = 0; i < n; ++i ) {
    for( j = 0; j < n; ++j ) {
      c[ i ][ j ] = a[ i ][ j ] + b[ i ][ j ];
    }
  }
  return( c );
}

double** sub( double **a, double **b, int n ) {
  double **c;
  int i, j;
  c = ( double** ) malloc( n * sizeof( double* ) );
  for( i = 0; i < n; i++ ) {
    c[ i ] = ( double* ) malloc( n * sizeof( double ) );
  }
  for( i = 0; i < n; ++i ) {
    for( j = 0; j < n; ++j ) {
      c[ i ][ j ] = a[ i ][ j ] - b[ i ][ j ];
    }
  }
  return( c );
}

void freeMatrix( double **mat, int n ) {
  for( int i = 0; i < n; i++ ) {
    free( mat[ i ] );
  }
  free( mat );
}


void strassenRec( double **a, double **b, double **c, int n ) {
  /*
   * Se a submatriz chegar à um tamanho definido pelo usuário
   * será realizada uma multiplicação convencional.
   */
  if( n <= BLOCK ) {
    multiplyBlock( a, b, c, n );
    return;
  }
  /*
   * Caso contrário, será feita a multiplicação de 4 submatrizes
   * de tamanho (n-1)x(n-1)
   */

  int m = n / 2;
  int i, j;
  double **a11, **a12, **a21, **a22;
  double **b11, **b12, **b21, **b22;
  a11 = ( double** ) malloc( m * sizeof( double* ) );
  a12 = ( double** ) malloc( m * sizeof( double* ) );
  a21 = ( double** ) malloc( m * sizeof( double* ) );
  a22 = ( double** ) malloc( m * sizeof( double* ) );
  b11 = ( double** ) malloc( m * sizeof( double* ) );
  b12 = ( double** ) malloc( m * sizeof( double* ) );
  b21 = ( double** ) malloc( m * sizeof( double* ) );
  b22 = ( double** ) malloc( m * sizeof( double* ) );
  for( i = 0; i < m; i++ ) {
    a11[ i ] = ( double* ) malloc( m * sizeof( double ) );
    a12[ i ] = ( double* ) malloc( m * sizeof( double ) );
    a21[ i ] = ( double* ) malloc( m * sizeof( double ) );
    a22[ i ] = ( double* ) malloc( m * sizeof( double ) );
    b11[ i ] = ( double* ) malloc( m * sizeof( double ) );
    b12[ i ] = ( double* ) malloc( m * sizeof( double ) );
    b21[ i ] = ( double* ) malloc( m * sizeof( double ) );
    b22[ i ] = ( double* ) malloc( m * sizeof( double ) );
    for( j = 0; j < m; j++ ) {
      a11[ i ][ j ] = a[ i ][ j ];
      a12[ i ][ j ] = a[ i ][ j + m ];
      a21[ i ][ j ] = a[ i + m ][ j ];
      a22[ i ][ j ] = a[ i + m ][ j + m ];

      b11[ i ][ j ] = b[ i ][ j ];
      b12[ i ][ j ] = b[ i ][ j + m ];
      b21[ i ][ j ] = b[ i + m ][ j ];
      b22[ i ][ j ] = b[ i + m ][ j + m ];
    }
  }
}

void strassen( double **a, double **b, double **c, int n ) {

}

int main( ) {
  int EventSet = PAPI_NULL;
  long long values[ 4 ], s, e;
  int retval;
  double **a, **b, **c;

  /* INICIALIZAÇÃO */

  PRINT( printf( "Inicializando Matriz: %dx%d\n", MATLEN, MATLEN ) );
  PRINT( printf( "Tamanho do bloco: %d\n", BLOCK ) );
  int i;

  a = ( double** ) malloc( MATLEN * sizeof( double* ) );
  b = ( double** ) malloc( MATLEN * sizeof( double* ) );
  c = ( double** ) malloc( MATLEN * sizeof( double* ) );
  for( i = 0; i < MATLEN; i++ ) {
    a[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
    b[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
    c[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
  }
  /*
   * CONFIGURAÇÃO DO PAPI
   * Init PAPI library
   */
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if( retval != PAPI_VER_CURRENT ) {
    printf( "Erro em PAPI_library_init : retval = %d\n", retval );
    exit( 1 );
  }
  if( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_create_eventset : retval = %d\n", retval );
    exit( 1 );
  }
  if( PAPI_add_event( EventSet, PAPI_L2_DCM ) != PAPI_OK ) {
    printf( "Erro em PAPI_L2_DCM\n" );
    exit( 1 );
  }
  if( PAPI_add_event( EventSet, PAPI_FP_OPS ) != PAPI_OK ) {
    printf( "Erro em PAPI_FP_INS\n" );
    exit( 1 );
  }
  if( PAPI_add_event( EventSet, PAPI_TOT_CYC ) != PAPI_OK ) {
    printf( "Erro em PAPI_TOT_CYC\n" );
    exit( 1 );
  }
  if( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK ) {
    printf( "Erro em PAPI_TOT_INS\n" );
    exit( 1 );
  }
  if( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_start" );
    exit( 1 );
  }
  s = PAPI_get_real_usec( );
  /* FUNÇÃO A SER AVALIADA */
  strassen( a, b, c );
  /* FIM DA FUNÇÃO A SER AVALIADA */
  e = PAPI_get_real_usec( );
  if( ( retval = PAPI_read( EventSet, &values[ 0 ] ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_read" );
    exit( 1 );
  }
  if( ( retval = PAPI_stop( EventSet, NULL ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_stop" );
    exit( 1 );
  }
  double cpi = ( double ) values[ 2 ] / ( double ) values[ 3 ];
  double icp = ( double ) values[ 3 ] / ( double ) values[ 2 ];
  /* double flops = ( double ) values[ 1 ]; */
  double mflops = ( double ) 2 * MATLEN * MATLEN * MATLEN;
  mflops = ( mflops / ( ( double ) ( e - s ) ) );
  /* EXIBINDO INFORMAÇÕES */
  PRINT(
    printf( "PAPI_L2_DCM = %lld\n", values[ 0 ] );
    printf( "PAPI_FP_OPS = %lld\n", values[ 1 ] );

    /* CPI */
    printf( "PAPI_TOT_CYC = %lld\n", values[ 2 ] );
    printf( "PAPI_TOT_INS = %lld\n", values[ 3 ] );
    printf( "CPI: %.2f\n", cpi );
    printf( "ICP: %.2f\n", icp );

    printf( "Wallclock time: %lld ms\n", e - s );
    printf( "MFLOPS: %g\n", mflops );
    printf( "Fim\n" );
    );
  /*       MAT BLk Time  DCM   MFLOPS CPI */
  printf( "%d, %d, %lld, %lld, %.2f, %.2f\n", MATLEN, BLOCK, e - s, values[ 0 ], mflops, cpi );
  return( 0 );
}
