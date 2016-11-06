#include "stats.h"
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef BOARDSIZE
#define BOARDSIZE 10
#endif

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

  n = BOARDSIZE;
  steps = 4 * ( n - 3 );
  n2 = n + 2;
  size = n2 * n2;
  /* ONLY HOST CODE */
  if( rank == 0 ) {
    data = ( int* ) malloc( size * sizeof( int ) );
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
    for( iRank = 1; iRank < commSize; ++iRank ) {
      int rows = floor( ( float ) n / commSize );
      int first = n2 + ( iRank ) * ( rows * n2 );
      if( iRank == commSize - 1 ) {
        rows = n - iRank * rows;
      }
      /*
       * printf( "iRank: %d, Rows: %d, First: %d, Size: %d, ", iRank, rows, first, size );
       * printf( "DataSize: %d\n", rows * n2 );
       * for(i = first; i < (first +  rows * n2); ++i){
       *   if( i % n2 == 0 ) {
       *     printf( "\n" );
       *   }
       *   printf( "%d ", data[ i ] );
       * }
       * printf( "\n" );
       */
      MPI_Send( &data[ first ], rows * n2, MPI_INTEGER, iRank, 42, MPI_COMM_WORLD );
    }
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
  if( rank == 0 ) { /* HOST */
    for( i = n2; i < ( ( rows + 1 ) * n2 ); ++i ) {
      localdata[ i ] = data[ i ];
    }
  }
  else { /* NOT HOST */
    MPI_Recv( &localdata[ n2 ], rows * n2, MPI_INTEGER, 0,
              42, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
  }
  // printf( "\n\nMy rank: %d, rows: %d, first: %d, localsize: %d, ", rank, rows, first, localSize );
  // printf( "DataSize: %d\n", rows * n2 );
  // printf( "\n" );
  // for( i = 0; i < localSize; ++i ) {
  //   if( i % n2 == 0 ) {
  //     printf( "\n  %d %02d: ", rank, i );
  //   }
  //   printf( "%d ", localdata[ i ] );
  // }
  // printf( "\n\n" );
  int ierr;
  MPI_Request request1, request2, request3, request4;
  /* inicializacao( ); */
  for( step = 0; step < steps; ++step ) {
    if( rank > 0 ) {
      MPI_Irecv( &localdata[ 0 ], n2, MPI_INTEGER, rank - 1, 10, MPI_COMM_WORLD, &request1 );
      MPI_Isend( &localdata[ n2 ], n2, MPI_INTEGER, rank - 1, 10, MPI_COMM_WORLD, &request2 );
    }
    if( rank < commSize - 1 ) {
      MPI_Irecv( &localdata[ localSize - n2 ], n2, MPI_INTEGER, rank + 1, 10, MPI_COMM_WORLD, &request3 );
      MPI_Isend( &localdata[ localSize - n2 * 2 ], n2, MPI_INTEGER, rank + 1, 10, MPI_COMM_WORLD, &request4 );
    }
    if( rank > 0 ) {
      MPI_Wait( &request1, MPI_STATUS_IGNORE );
      MPI_Wait( &request2, MPI_STATUS_IGNORE );
    }
    if( rank < commSize - 1 ) {
      MPI_Wait( &request3, MPI_STATUS_IGNORE );
      MPI_Wait( &request4, MPI_STATUS_IGNORE );
    }
    // for( i = 0; i < localSize; ++i ) {
    //   if( i % n2 == 0 ) {
    //     printf( "\n  %d %02d: ", rank, i );
    //   }
    //   printf( "%d ", localdata[ i ] );
    // }
    // printf( "\n\n" );
    evolve( localdata, aux, n2, rows + 2 );
    aux2 = localdata; localdata = aux; aux = aux2;
    // for( i = n2; i < localSize - n2; ++i ) {
    //   if( i % n2 == 0 ) {
    //     printf( "\n  %d %02d: ", rank, i );
    //   }
    //   printf( "%d ", localdata[ i ] );
    // }
    // printf( "\n\n" );
  }
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank != 0 ) {
    MPI_Send( &localdata[ n2 ], rows * n2, MPI_INTEGER, 0, 42, MPI_COMM_WORLD );
  }else{
    for( i = n2; i < ( ( rows + 1 ) * n2 ); ++i ) {
      data[ i ] = localdata[ i ];
    }
    for(iRank = 1; iRank < commSize; ++iRank){
      int rows = floor( ( float ) n / commSize );
      int first = n2 + ( iRank ) * ( rows * n2 );
      if( iRank == commSize - 1 ) {
        rows = n - iRank * rows;
      }
      MPI_Recv( &data[ first ], rows * n2, MPI_INTEGER, iRank, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
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
  MPI_Finalize( );
  return( 0 );
}
