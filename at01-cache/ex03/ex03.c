#include <stdio.h>
#include <stdlib.h>

#ifndef MATLEN
#define MATLEN 128
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


double** multiplyBlock( double **a, double **b, int n ) {
  int i, j, k;
  double **c = ( double** ) malloc( n * sizeof( double* ) );
  for( i = 0; i < n; ++i ) {
    c[ i ] = ( double* ) malloc( n * sizeof( double ) );
    for( k = 0; k < n; ++k ) {
      for( j = 0; j < n; ++j ) {
        c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
      }
    }
  }
  return( c );
}

double** add( double **a, double **b, int n ) {
  int i, j;
  double **c = ( double** ) malloc( n * sizeof( double* ) );
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
  int i, j;
  double **c = ( double** ) malloc( n * sizeof( double* ) );
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

void add2( double **a, double **b, int n ) {
  int i, j;
  for( i = 0; i < n; ++i ) {
    for( j = 0; j < n; ++j ) {
      a[ i ][ j ] += b[ i ][ j ];
    }
  }
}

void sub2( double **a, double **b, int n ) {
  int i, j;
  for( i = 0; i < n; ++i ) {
    for( j = 0; j < n; ++j ) {
      a[ i ][ j ] -= b[ i ][ j ];
    }
  }
}

void freeMatrix( double **mat, int n ) {
  for( int i = 0; i < n; i++ ) {
    free( mat[ i ] );
  }
  free( mat );
}


double** strassen( double **a, double **b, int n ) {
  /*
   * Se a submatriz chegar à um tamanho definido pelo usuário
   * será realizada uma multiplicação convencional.
   */
  if( n <= BLOCK ) {
    return( multiplyBlock( a, b, n ) );
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
  /* Calculando produtos temporários q1->q7, já liberando as somas
   * temporarias t1->t10*/
  double **t1 = add( a11, a22, m );
  double **t6 = add( b11, b22, m );
  double **q1 = strassen( t1, t6, m );
  freeMatrix( t1, m );
  freeMatrix( t6, m );

  double **t2 = add( a21, a22, m );
  double **q2 = strassen( t2, b11, m );
  freeMatrix( t2, m );

  double **t7 = sub( b12, b22, m );
  double **q3 = strassen( a11, t7, m );
  freeMatrix( t7, m );

  double **t8 = sub( b21, b11, m );
  double **q4 = strassen( a22, t8, m );
  freeMatrix( t8, m );

  double **t3 = add( a11, a12, m );
  double **q5 = strassen( t3, b22, m );
  freeMatrix( t3, m );

  double **t4 = sub( a21, a11, m );
  double **t9 = add( b11, b12, m );
  double **q6 = strassen( t4, t9, m );
  freeMatrix( t4, m );
  freeMatrix( t9, m );

  double **t5 = sub( a12, a22, m );
  double **t10 = add( b21, b22, m );
  double **q7 = strassen( t5, t10, m );
  freeMatrix( t5, m );
  freeMatrix( t10, m );

  /* Calculando c11 c12 c21 e c22 */
  double **c11 = add( q1, q4, m ); /* c11 = q1 + q4 */
  sub2( c11, q5, m ); /* c11 = c11 - q5; */
  add2( c11, q7, m ); /* c11 = c11 + q7; */

  double **c12 = add( q3, q5, m ); /* c12 = q3 + q5 */

  double **c21 = add( q2, q4, m ); /* c21 = q2 + q4 */

  double **c22 = sub( q1, q2, m ); /* c22 = q1 - q2 */
  add2( c22, q3, m ); /* c22 = c22 + q3; */
  add2( c22, q6, m ); /* c22 = c22 + q6; */

  freeMatrix( q1, m );
  freeMatrix( q2, m );
  freeMatrix( q3, m );
  freeMatrix( q4, m );
  freeMatrix( q5, m );
  freeMatrix( q6, m );
  freeMatrix( q7, m );

  double **c = ( double** ) malloc( n * sizeof( double* ) );
  for( i = 0; i < n; i++ ) {
    c[ i ] = ( double* ) malloc( n * sizeof( double ) );
  }
  for( i = 0; i < m; i++ ) {
    for( j = 0; j < m; j++ ) {
      c[ i ][ j ] = c11[ i ][ j ];
      c[ i ][ j + m ] = c12[ i ][ j ];
      c[ i + m ][ j ] = c21[ i ][ j ];
      c[ i + m ][ j + m ] = c22[ i ][ j ];
    }
  }
  freeMatrix( c11, m );
  freeMatrix( c12, m );
  freeMatrix( c21, m );
  freeMatrix( c22, m );
  return( c );
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
  for( i = 0; i < MATLEN; i++ ) {
    a[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
    b[ i ] = ( double* ) malloc( MATLEN * sizeof( double ) );
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
  c = strassen( a, b, MATLEN );
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
  freeMatrix( a, MATLEN );
  freeMatrix( b, MATLEN );
  freeMatrix( c, MATLEN );
  return( 0 );
}
