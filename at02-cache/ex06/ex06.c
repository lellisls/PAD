#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LENGTH
#define LENGTH 1024
#endif

#ifndef QUIET
#define PRINT( exp ) exp
#else
#define PRINT( exp )
#endif

#ifdef MODE1
  #define MODE 1
#endif
#ifdef MODE2
  #define MODE 2
#endif
#ifdef MODE3
  #define MODE 3
#endif

typedef struct {
  double a, b, c;
} est_abc;

#include "papi.h"

void freeMatrix( double **mat ) {
  for( int i = 0; i < LENGTH; i++ ) {
    free( mat[ i ] );
  }
  free( mat );
}

int main( ) {
  int EventSet = PAPI_NULL;
  long long values[ 4 ], s, e;
  int retval;

  /* INICIALIZAÇÃO */

  int i;
  PRINT( printf( "Inicializando Vetor: %d\n", LENGTH ) );
#ifdef MODE1
  double **abc;
  abc = ( double** ) malloc( LENGTH * sizeof( double* ) );
  for( i = 0; i < LENGTH; ++i ) {
    abc[ i ] = ( double* ) malloc( 3 * sizeof( double ) );
  }
#endif
#ifdef MODE2
  double **abc;
  abc = ( double** ) malloc( 3 * sizeof( double* ) );
  for( i = 0; i < 3; ++i ) {
    abc[ i ] = ( double* ) malloc( LENGTH * sizeof( double ) );
  }
#endif
#ifdef MODE3
  est_abc *abc;
  abc = ( est_abc* ) malloc( LENGTH * sizeof( est_abc ) );
#endif

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
  if( PAPI_add_event( EventSet, PAPI_DP_OPS ) != PAPI_OK ) {
    printf( "Erro em PAPI_DP_OPS\n" );
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
  /*
   * for(i = 0; i < LENGTH; ++i){
   *   b[i] = sin(a[i]*2.0) + 1.0;
   *   c[i] = cos(b[i]+a[i]) + 4.0;
   * }
   */
#ifdef MODE1
  for( i = 0; i < LENGTH; ++i ) {
    abc[ i ][ 1 ] = sin( abc[ i ][ 0 ] * 2.0 ) + 1.0;
    abc[ i ][ 2 ] = cos( abc[ i ][ 1 ] + abc[ i ][ 0 ] ) + 4.0;
  }
#endif
#ifdef MODE2
  for( i = 0; i < LENGTH; ++i ) {
    abc[ 1 ][ i ] = sin( abc[ 0 ][ i ] * 2.0 ) + 1.0;
    abc[ 2 ][ i ] = cos( abc[ 1 ][ i ] + abc[ 0 ][ i ] ) + 4.0;
  }
#endif
#ifdef MODE3
  for( i = 0; i < LENGTH; ++i ) {
    abc[ i ].b = sin( abc[ i ].a * 2.0 ) + 1.0;
    abc[ i ].c = cos( abc[ i ].b + abc[ i ].a ) + 4.0;
  }
#endif
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
  /* double mflops = ( double ) 2 * LENGTH * LENGTH * LENGTH; */
  mflops = ( mflops / ( ( double ) ( e - s ) ) );
  /* EXIBINDO INFORMAÇÕES */
  PRINT(
    printf( "PAPI_L2_DCM = %lld\n", values[ 0 ] );
    printf( "PAPI_DP_OPS = %lld\n", values[ 1 ] );

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
  printf( "%d, %d, %lld, %lld, %.2f, %.2f\n", LENGTH, MODE, e - s, values[ 0 ], mflops, cpi );
  return( 0 );
}
