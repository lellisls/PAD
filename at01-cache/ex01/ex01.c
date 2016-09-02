#include <stdio.h>
#include <stdlib.h>

#ifndef MATLEN
#define MATLEN 1024
#endif

#ifndef MODE
#define MODE 1
#endif

#ifndef QUIET
#define PRINT( exp ) exp
#else
#define PRINT( exp )
#endif

#include "papi.h"

void freeMatrix( double **mat ) {
  for( int i = 0; i < MATLEN; i++ ) {
    free( mat[ i ] );
  }
  free( mat );
}

int main( ) {
  int EventSet = PAPI_NULL;
  long long values[ 4 ], s, e;
  int retval;

  /* INICIALIZAÇÃO */

  int i, j, k;
  double **a, **b, **c;
  PRINT( printf( "Inicializando Matriz: %dx%d\n", MATLEN, MATLEN ) );
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
  switch( MODE ) {
      case 0: { /* IJK */
      for( i = 0; i < MATLEN; ++i ) {
        for( k = 0; k < MATLEN; ++k ) {
          for( j = 0; j < MATLEN; ++j ) {
            c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
          }
        }
      }
      break;
    }
      case 1: { /* IKJ */
      for( i = 0; i < MATLEN; ++i ) {
        for( k = 0; k < MATLEN; ++k ) {
          for( j = 0; j < MATLEN; ++j ) {
            c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
          }
        }
      }
      break;
    }
      case 2: { /* JIK */
      for( j = 0; j < MATLEN; ++j ) {
        for( i = 0; i < MATLEN; ++i ) {
          for( k = 0; k < MATLEN; ++k ) {
            c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
          }
        }
      }
      break;
    }
      case 3: { /* JKI */
      for( j = 0; j < MATLEN; ++j ) {
        for( k = 0; k < MATLEN; ++k ) {
          for( i = 0; i < MATLEN; ++i ) {
            c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
          }
        }
      }
      break;
    }
      case 4: { /* KIJ */
      for( k = 0; k < MATLEN; ++k ) {
        for( i = 0; i < MATLEN; ++i ) {
          for( j = 0; j < MATLEN; ++j ) {
            c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
          }
        }
      }
      break;
    }
      case 5: { /* KJI */
      for( k = 0; k < MATLEN; ++k ) {
        for( j = 0; j < MATLEN; ++j ) {
          for( i = 0; i < MATLEN; ++i ) {
            c[ i ][ j ] = c[ i ][ j ] + a[ i ][ k ] * b[ k ][ j ];
          }
        }
      }
      break;
    }
  }
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
  double mflops = ( double ) values[ 1 ];
  // double mflops = ( double ) 2 * MATLEN * MATLEN * MATLEN;
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
  /*      Time  DCM   MFLOPS CPI */
  printf( "%d, %d, %lld, %lld, %.2f, %.2f\n", MATLEN, MODE, e - s, values[ 0 ], mflops, cpi );
  freeMatrix(a);
  freeMatrix(b);
  freeMatrix(c);
  return( 0 );
}
