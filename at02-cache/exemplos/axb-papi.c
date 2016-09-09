#include <stdio.h>

#include "papi.h"

#define MATLEN 10000

int main() {

  int i, j;
  int EventSet = PAPI_NULL;
  long long values[3], s, e;
  int retval;
  double **a, *y, *x;

  a = (double **) malloc(MATLEN*sizeof(double *));

  for(i=0; i<MATLEN; i++) {
    a[i] = (double *) malloc(MATLEN*sizeof(double));
  }

  x = (double *) malloc(MATLEN*sizeof(double));
  y = (double *) malloc(MATLEN*sizeof(double));


  /* Init PAPI library */
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if ( retval != PAPI_VER_CURRENT ) {
    printf("Erro em PAPI_library_init : retval = %d\n", retval);
    exit (1);
  }

  if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK ) {
    printf("Erro em PAPI_create_eventset : retval = %d\n", retval);
    exit (1);
  }

  if (PAPI_add_event(EventSet, PAPI_L2_DCM) != PAPI_OK) {
    printf("Erro em PAPI_L2_DCM\n");
    exit (1);
  }

  /* if (PAPI_add_event(EventSet, PAPI_L3_DCM) != PAPI_OK) { */
  /*   printf("Erro em PAPI_L3_DCM\n"); */
  /*   exit (1); */
  /* } */

  if (PAPI_add_event(EventSet, PAPI_FP_OPS) != PAPI_OK) {
    printf("Erro em PAPI_FP_INS\n");
    exit (1);
  }

  if ( ( retval = PAPI_start( EventSet ) ) != PAPI_OK ) {
    printf("Erro em PAPI_start");
    exit (1);
  }

  s = PAPI_get_real_usec();

  for(i=0; i<MATLEN; i++)
    for(j=0; j<MATLEN; j++)
      y[i] = y[i] + a[i][j]*x[j];

  /* for(j=0; j<MATLEN; j++) */
  /*   for(i=0; i<MATLEN; i++) */
  /*     y[i] = y[i] + a[i][j]*x[j]; */

  e = PAPI_get_real_usec();

  if ( ( retval = PAPI_read( EventSet, &values[0] ) ) != PAPI_OK ) {
    printf("Erro em PAPI_read");
    exit (1);
  }

  if ( ( retval = PAPI_stop( EventSet, NULL ) ) != PAPI_OK ) {
    printf("Erro em PAPI_stop");
    exit (1);
  }

  printf("PAPI_L2_DCM = %lld\n", values[0]);
  printf("PAPI_FP_OPS = %lld\n", values[1]);
  /* printf("PAPI_L3_DCM = %lld\n", values[2]); */

  printf("Wallclock time: %lld ms\n",e-s);

  printf("MFLOPS: %g\n", (200000000./((double)(e-s))));

  printf("Fim\n");
  return(0);
}
