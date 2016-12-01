#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>

int main( void ) {

  float * data;
  long int n = 1000000;

  data = (float *) malloc( n * sizeof( float ) );
  double start = omp_get_wtime();
  int i;
  for( i = 0; i < n; ++i){
    data[ i ] = (float) rand() / 1000.0f;
  }
  float total = 0.;
  for( i = 0; i < n; ++i){
    total += data[i];
  }

  printf( "\tTempo Total  : %f \n", omp_get_wtime( ) - start );
  printf( "Somatorio = %f\n", total );

  return 0;
}
