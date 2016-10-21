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

void OddEven( int a[], int n ) {
  int i, j, nHalf, lastEven, lastOdd;

  nHalf = n / 2;
  if( n % 2 == 0 ) {
    lastEven = nHalf - 1;
    lastOdd = nHalf - 2;
  }
  else {
    lastEven = nHalf - 1;
    lastOdd = nHalf - 1;
  }
  #pragma omp parallel for default(none) shared(a) private(i, j) firstprivate(lastOdd, lastEven, n)
  for( i = 0; i < n - 1; i++ ) {
    for( j = 0; j <= lastOdd; j++ ) {
      ce( &a[ 2 * j + 1 ], &a[ 2 * j + 2 ] ); /* odd */
    }
    for( j = 0; j <= lastEven; j++ ) {
      ce( &a[ 2 * j ], &a[ 2 * j + 1 ] ); /* even */
    }
  }
}


int main( int argc, char **argv ) {
  int num_thds = 1, size = 1024, *a, i;
  char str[ 20 ];
  if( argc == 3 ) {
    size = atoi( argv[ 1 ] );
    num_thds = atoi( argv[ 2 ] );
  }
  else {
    printf( "usage: <size> <num_thds>\n" );
    return( 0 );
  }
  omp_set_num_threads( num_thds );
  /* INICIALIZAÇÃO */

  a = ( int* ) malloc( size * sizeof( int ) );
  srand( 424242 );
  for( i = 0; i < size; ++i ) {
    a[ i ] = rand( ) % size;
    PRINT( printf( "%d ", a[ i ] ) );
  }
  PRINT( printf( "\n" ) );

  inicializacao( );

  OddEven( a, size );

  sprintf( str, "%d; %d", size );
  avaliacao( str, size );
  PRINT(
  for( i = 0; i < size; ++i ) {
    printf( "%d ", a[ i ] );
  }
  printf( "\n" );
  );

  free( a );
  return( 0 );
}
