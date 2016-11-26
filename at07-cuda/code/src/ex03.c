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
    data[ i ] = rand() / 1000.0f;
  }

  printf( "\tTempo Total  : %f \n", omp_get_wtime( ) - start );

}
