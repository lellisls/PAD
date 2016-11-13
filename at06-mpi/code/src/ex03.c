#include "stats.h"
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BOARDSIZE
#define BOARDSIZE 1000
#endif

void evolve( int *val, int *aux, int width, int heigth ) {
  int up, upright, right, rightdown, down, downleft, left, leftup;
  int sum = 0, estado, i, j, inj;
  #pragma omp parallel for private(i, j, inj, estado, sum, up, upright, right, rightdown, down, downleft, left, leftup) shared(val, aux)
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
  int n, n2, steps, size, commSize, rank, provided;
  int i, step, *data, iRank;
  MPI_Win datawin, upgzwin, lowgzwin;
  MPI_Status status;
  MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  MPI_Comm_size( MPI_COMM_WORLD, &commSize );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  n = BOARDSIZE;
  steps = 4 * ( n - 3 );
  n2 = n + 2;
  size = n2 * n2;
  /* ONLY HOST CODE */
  if( rank == 0 ) {
    MPI_Alloc_mem( size * sizeof( int ), MPI_INFO_NULL, &data );
    for( i = 0; i < size; ++i ) {
      data[ i ] = 0;
      /* aux[ i ] = 0; */
    }
    data[ 1 * n2 + 2 ] = 1;
    data[ 2 * n2 + 3 ] = 1;
    data[ 3 * n2 + 1 ] = 1;
    data[ 3 * n2 + 2 ] = 1;
    data[ 3 * n2 + 3 ] = 1;

    PRINT( print( data, n2, 0, 12 ); );

    /*
     * printf( "\n" );
     * for( i = 0; i < size; ++i ) {
     *   if( i % n2 == 0 ) {
     *     printf( "\n  %d %03d: ", rank, i );
     *   }
     *   printf( "%d ", data[ i ] );
     * }
     * printf( "\n\n" );
     */

    MPI_Win_create( data, size, sizeof( int ), MPI_INFO_NULL, MPI_COMM_WORLD, &datawin );
  }
  else {
    MPI_Win_create( MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &datawin );
  }
  /* END OF HOST CODE */
  int rows = floor( ( float ) n / commSize );
  int first = n2 + ( rank ) * ( rows * n2 );
  if( rank == commSize - 1 ) {
    rows = n - rank * rows;
  }
  int localSize = ( rows + 2 ) * n2;
  int *localdata = ( int* ) calloc( localSize, sizeof( int ) );
  int *aux = ( int* ) calloc( localSize, sizeof( int ) );
  int *aux2;

  MPI_Win_fence( 0, datawin );
  MPI_Get( &localdata[ n2 ], rows * n2, MPI_INT, 0, first, rows * n2, MPI_INT, datawin );
  MPI_Win_fence( 0, datawin );

  /*
   * printf( "\n\nMy rank: %d, rows: %d, first: %d, localsize: %d, DataSize: %d\n", rank, rows, first, localSize, rows *
   * n2 );
   * printf( "\n" );
   * for( i = 0; i < localSize; ++i ) {
   *   if( i % n2 == 0 ) {
   *     printf( "\n  %d %02d: ", rank, i );
   *   }
   *   printf( "%d ", localdata[ i ] );
   * }
   * printf( "\n\n" );
   */

  MPI_Win_create( &localdata[ localSize - n2 * 2 ], n2, sizeof( int ), MPI_INFO_NULL, MPI_COMM_WORLD, &upgzwin );
  MPI_Win_create( &localdata[ n2 ], n2, sizeof( int ), MPI_INFO_NULL, MPI_COMM_WORLD, &lowgzwin );

  int ierr;
  MPI_Request request1, request2, request3, request4;
  /* inicializacao( ); */
  for( step = 0; step < steps; ++step ) {
    /* MPI_Barrier(MPI_COMM_WORLD); */
    MPI_Win_fence( 0, upgzwin );
    MPI_Win_fence( 0, lowgzwin );
    if( rank > 0 ) {
      MPI_Get( &localdata[ 0 ], n2, MPI_INT, rank - 1, 0, n2, MPI_INT, upgzwin );
    }
    if( rank < commSize - 1 ) {
      MPI_Get( &localdata[ localSize - n2 ], n2, MPI_INT, rank + 1, 0, n2, MPI_INT, lowgzwin );
    }
    MPI_Win_fence( 0, upgzwin );
    MPI_Win_fence( 0, lowgzwin );

    evolve( localdata, aux, n2, rows + 2 );
    for( i = 0; i < localSize; ++i ) {
      localdata[ i ] = aux[ i ];
    }
    /* aux2 = localdata; localdata = aux; aux = aux2; */
  }
  /* */
  MPI_Win_fence( 0, datawin );
  MPI_Put( &localdata[ n2 ], rows * n2, MPI_INT, 0, first, rows * n2, MPI_INT, datawin );
  MPI_Win_fence( 0, datawin );
  if( rank == 0 ) {
    PRINT( print( data, n2, n2 - 12, n2 ); );
  }
  /* char str[ 20 ]; */
  /*
   * sprintf( str, "%d; %d", n );
   * avaliacao( str, size );
   * if( rank == 0){
   *
   * }
   */
  MPI_Win_free( &datawin );
  MPI_Finalize( );
  return( 0 );
}
