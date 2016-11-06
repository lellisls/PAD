#include "stats.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void evolve( int *val, int *aux, int width, int heigth ) {
  int up, upright, right, rightdown, down, downleft, left, leftup;
  int sum = 0, estado, i, j, inj;
/* #pragma omp parallel for default(none) private(i, j, inj, estado, sum, up, upright, right, rightdown, down, downleft,
 * left, leftup) shared(val, aux) firstprivate(n) */
  for( i = 1; i < heigth - 1; ++i ) {
    for( j = 1; j < width - 1; ++j ) {
      inj = i * width + j;
      estado = val[ inj ];

      up = val[ inj - width ];
      upright = val[ inj - width + 1 ];
      right = val[ inj + 1 ];
      rightdown = val[ inj + width + 1 ];
      down = val[ inj + width ];
      downleft = val[ inj + width - 1 ];
      left = val[ inj - 1 ];
      leftup = val[ inj - width - 1 ];

      sum = up + upright + right + rightdown + down + downleft + left + leftup;
      if( sum == 3 ) {
        estado = 1;
      }
      else if( ( estado == 1 ) && ( ( sum < 2 ) || ( sum > 3 ) ) ) {
        estado = 0;
      }
      aux[ inj ] = estado;
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

int main( int argc, char *argv[] ) {
  int n, n2, steps, size, commSize, rank;
  int i, step, *data, iRank;

  MPI_Status status;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &commSize );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  n = 10;
  steps = 4 * ( n - 3 );
  n2 = n + 2;
  size = n2 * n2;
  if( rank == 0 ) {
    data = ( int* ) malloc( size * sizeof( int ) );
    for( i = 0; i < size; ++i ) {
      data[ i ] = 0;
      /* aux[ i ] = 0; */
    }
    data[ 1 * n + 2 ] = 1;
    data[ 2 * n + 3 ] = 1;
    data[ 3 * n + 1 ] = 1;
    data[ 3 * n + 2 ] = 1;
    data[ 3 * n + 3 ] = 1;
    PRINT( print( data, n, 0, 12 ); );
    for( iRank = 1; iRank < commSize; ++iRank ) {
      int rows = floor( ( float ) size / commSize );
      int first = n2 + iRank * ( rows * n2 );
      if( iRank == commSize - 1 ) {
        rows = size - first;
      }
      printf("Size: %d\n", rows * n2);
      MPI_Send( &data[ first ], rows * n2, MPI_INTEGER, iRank, 42, MPI_COMM_WORLD );
    }
  }
  int localHeigth = floor( ( float ) size / commSize ) + 2;
  int first = rank * localHeigth;
  if( rank == commSize - 1 ) {
    localHeigth = size - first + 2;
  }
  int localSize = localHeigth * n2;
  int *localdata = ( int* ) malloc( sizeof( int ) * localSize );
  int *aux = ( int* ) malloc( sizeof( int ) * localSize );
  int *aux2;

  for( i = 0; i < localSize; ++i ) {
    localdata[ i ] = 0;
  }
  if( rank == 0 ) {
    for( i = n2; i < localSize; ++i ) {
      localdata[ i ] = data[ n2 + i ];
    }
  }
  else {
    MPI_Recv( &localdata[ n2 ], ( localHeigth - 2 ) * n2, MPI_INTEGER, 0,
              42, MPI_COMM_WORLD, &status );
  }
  int ierr;
  MPI_Request request1, request2, request3, request4;
  /* inicializacao( ); */
  for( step = 0; step < steps; ++step ) {
    if( rank > 0 ) {
      MPI_Irecv( &localdata[ 0 ], n2, MPI_INTEGER, rank - 1, 10, MPI_COMM_WORLD, &request1 );
      MPI_Isend( &localdata[ n2 ], n2, MPI_INTEGER, rank - 1, 10, MPI_COMM_WORLD, &request2 );
    }
    if( rank < commSize - 1 ) {
      MPI_Irecv( &localdata[ localSize - n2 ], n2, MPI_INTEGER, rank + 1, 11, MPI_COMM_WORLD, &request3 );
      MPI_Isend( &localdata[ localSize - n2 * 2 ], n2, MPI_INTEGER, rank + 1, 11, MPI_COMM_WORLD, &request4 );
    }
    if( rank > 0 ) {
      MPI_Wait( &request1, &status );
      MPI_Wait( &request2, &status );
    }
    if( rank < commSize - 1 ) {
      MPI_Wait( &request3, &status );
      MPI_Wait( &request4, &status );
    }
    evolve( localdata, aux, n2, localHeigth );
    aux2 = localdata; localdata = aux; aux = aux2;
  }
  /* char str[ 20 ]; */
  /*
   * sprintf( str, "%d; %d", n );
   * avaliacao( str, size );
   * if( rank == 0){
   * PRINT( print( data, n, n - 12, n ); );
   * }
   */
  MPI_Finalize( );
  return( 0 );
}
