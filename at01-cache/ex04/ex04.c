#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <papi.h>

#ifndef MATLEN
#define MATLEN 128
#endif

#ifndef QUIET
#define PRINT( exp ) exp
#else
#define PRINT( exp )
#endif


int main( ) {
  int EventSet = PAPI_NULL;
  long long values[ 4 ], s, e;
  int retval;
  double *a, *b, *c;

  /* INICIALIZAÇÃO */

  PRINT( printf( "Inicializando Matriz: %dx%d\n", MATLEN, MATLEN ) );

  a = ( double* ) malloc( MATLEN * MATLEN * sizeof( double ) );
  b = ( double* ) malloc( MATLEN * MATLEN * sizeof( double ) );
  c = ( double* ) malloc( MATLEN * MATLEN * sizeof( double ) );

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

  cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
               MATLEN, MATLEN, MATLEN, 1.0, a, MATLEN,
               b, MATLEN, 0.0, c, MATLEN );

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
  printf( "%d, %lld, %lld, %.2f, %.2f\n", MATLEN, e - s, values[ 0 ], mflops, cpi );
  free( a );
  free( b );
  free( c );
  return( 0 );
}
