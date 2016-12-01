#include <omp.h>

#include "stdio.h"
#include "stdlib.h"

#define SIZE 1000

void evolve( int *val, int *aux, int n ) {
  int up, upright, right, rightdown, down, downleft, left, leftup;
  int sum = 0, estado, i, j, inj;
  #pragma \
    omp parallel for default(none) private(i, j, inj, estado, sum, up, upright, right, rightdown, down, downleft, left, leftup) shared(val, aux) firstprivate(n)
  for( i = 1; i < n - 1; ++i ) {
    for( j = 1; j < n - 1; ++j ) {
      estado = val[ i * n + j ];
      up = val[ ( i - 1 ) * n + j ];
      upright = val[ ( i - 1 ) * n + j + 1 ];
      right = val[ i * n + j + 1 ];
      rightdown = val[ ( i + 1 ) * n + j + 1 ];
      down = val[ ( i + 1 ) * n + j ];
      downleft = val[ ( i + 1 ) * n + j - 1 ];
      left = val[ i * n + j - 1 ];
      leftup = val[ ( i - 1 ) * n + j - 1 ];
      sum = up + upright + right + rightdown + down + downleft + left + leftup;
      if( sum == 3 ) {
        estado = 1;
      }
      else if( ( estado == 1 ) && ( ( sum < 2 ) || ( sum > 3 ) ) ) {
        estado = 0;
      }
      aux[ i * n + j ] = estado;
    }
  }
}

void print( int *val, int n, int start, int end ) {
  int i, j;
  printf( "+" );
  for( i = start; i < end; ++i ) {
    printf( "-" );
  }
  printf( "+\n" );
  for( i = start; i < end; ++i ) {
    printf( "|" );
    for( j = start; j < end; ++j ) {
      if( val[ i * n + j ] ) {
        printf( "0" );
      }
      else {
        printf( " " );
      }
    }
    printf( "|\n" );
  }
  printf( "+" );
  for( i = start; i < end; ++i ) {
    printf( "-" );
  }
  printf( "+\n" );
}

int main( int argc, char const *argv[] ) {
  int n = SIZE, steps, size;
  if( argc > 1 )
    n = atoi( argv[ 1 ] );
  size = n * n;
  if( argc < 3 ) {
    steps = 4 * ( n - 5 );
  }
  else {
    steps = atoi( argv[ 2 ] );
  }
  int *data = ( int* ) malloc( size * sizeof( int ) );
  int *aux = ( int* ) malloc( size * sizeof( int ) );
  int *aux2;
  int i, step;
  for( i = 0; i < size; ++i ) {
    data[ i ] = 0;
    aux[ i ] = 0;
  }
  data[ 1 * n + 2 ] = 1;
  data[ 2 * n + 3 ] = 1;
  data[ 3 * n + 1 ] = 1;
  data[ 3 * n + 2 ] = 1;
  data[ 3 * n + 3 ] = 1;
  double start = omp_get_wtime();
  print( data, n, 0, 12 );
  for( step = 0; step < steps; ++step ) {
    evolve( data, aux, n );
    aux2 = data; data = aux; aux = aux2;

    // PRINT(print( data, n, 0, 12 ););
  }
  print( data, n, n - 12, n );
  printf( "\tTempo total : \t %f \n", omp_get_wtime( ) - start );
  return( 0 );
}
