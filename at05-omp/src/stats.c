#include "stats.h"
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

int EventSet = PAPI_NULL;
long long values[ 4 ], s, e;

void inicializacao( ) {
  #ifndef DISABLESTATS
  /* INICIALIZAÇÃO */
  int retval;
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

  if( PAPI_add_event( EventSet, PAPI_SP_OPS ) != PAPI_OK ) {
    printf( "Erro em PAPI_SP_OPS\n" );
    exit( 1 );
  }
  // // if( PAPI_add_event( EventSet, PAPI_TOT_CYC ) != PAPI_OK ) {
  // //   printf( "Erro em PAPI_TOT_CYC\n" );
  // //   exit( 1 );
  // // }
  // // if( PAPI_add_event( EventSet, PAPI_TOT_INS ) != PAPI_OK ) {
  // //   printf( "Erro em PAPI_TOT_INS\n" );
  // //   exit( 1 );
  // // }
  if( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_start" );
    exit( 1 );
  }
  s = PAPI_get_real_usec( );
  #endif
}

void avaliacao( char *text, int size ){
  #ifndef DISABLESTATS
  int retval;
  /* FIM DA FUNÇÃO A SER AVALIADA */
  e = PAPI_get_real_usec( );
  if( ( retval = PAPI_read( EventSet, &values[ 0 ] ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_read.\n" );
    exit( 1 );
  }
  if( ( retval = PAPI_stop( EventSet, NULL ) ) != PAPI_OK ) {
    printf( "Erro em PAPI_stop.\n" );
    exit( 1 );
  }
  /*
   * double cpi = ( double ) values[ 2 ] / ( double ) values[ 3 ];
   * double icp = ( double ) values[ 3 ] / ( double ) values[ 2 ];
   */
  // double cpe = ( double ) values[ 2 ] / size;
  double mflops = ( double ) values[ 1 ];
  mflops = ( mflops / ( ( double ) ( e - s ) ) );
  double et = (e - s)/1000000.0 ;
  /* EXIBINDO INFORMAÇÕES */
  printf( "%s; %.4f; %lld\n", text, et, values[ 0 ]);

  PRINT(
    printf( "PAPI_L2_DCM = %lld\n", values[ 0 ] );
    printf( "PAPI_SP_OPS = %lld\n", values[ 1 ] );

    /* CPI */
    // printf( "PAPI_TOT_CYC = %lld\n", values[ 2 ] );
    // printf( "PAPI_TOT_INS = %lld\n", values[ 3 ] );
    // printf( "CPE: %.2f\n", cpe );

    printf( "Wallclock time: %.4f s\n", et );
    // printf( "MFLOPS: %g\n", mflops );
    // printf( "Fim\n" );
    );

    #endif
}
