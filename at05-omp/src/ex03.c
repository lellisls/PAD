#include "stats.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void ce( int *a, int *b ) {
  int t;
  if( *a > *b ) {
    t = *a; *a = *b; *b = t;
  }
}

void CountNumbers( int a[], int n ) {
  int sum = 0;
#pragma omp parallel default(none ) reduction(+:sum) shared(a,n)
  {
    int *count = ( int* ) calloc( 1000, sizeof( int ) );
    int i;
#pragma omp for
    for( i = 0; i < n; i++ ) {
      count[ a[ i ] ]++;
    }
    sum = 0;
    for( i = 0; i < 1000; i++ ) {
      sum += count[ i ];
    }
    free( count );
  }
  volatile int ssssss = sum;
  printf( "%d\n", sum );
}


int main( int argc, char **argv ) {
  int num_thds = 1, size = 100000, *a, i;
  char str[ 20 ];
  if( argc == 3 ) {
    size = atoi( argv[ 1 ] );
    num_thds = atoi( argv[ 2 ] );
  }
  else {
    printf( "usage: <size> <num_thds>\n" );
    return( 0 );
  }
  /* INICIALIZAÇÃO */
  omp_set_num_threads(num_thds);
  a = ( int* ) malloc( size * sizeof( int ) );
  srand( 424242 );
  for( i = 0; i < size; ++i ) {
    a[ i ] = rand( ) % 1000;
  }
  inicializacao( );

  CountNumbers( a, size );

  sprintf( str, "%d; %d", size, num_thds );
  avaliacao( str, size );

  free( a );
  return( 0 );
}
