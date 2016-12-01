#include <cuda.h>
#include <omp.h>

/* #include "stats.h" */
#include "stdio.h"
#include "stdlib.h"
#define BSIZE 16
#define SIZE 1000


__global__ void Evolve( int *val, int *aux, int n ) {
  int up, upright, right, rightdown, down, downleft, left, leftup;
  int sum = 0, estado;
  const int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int b2 = BSIZE + 2;
  __shared__ float sdata[ b2 ][ b2 ];

  sdata[ ty ][ tx ] = val[ i * n + j ];
  if( ( tx == 1 ) && ( ty == 1 ) ) {
    int stx = blockIdx.x * blockDim.x;
    int sty = blockIdx.y * blockDim.y;
    sdata[ 0      ][ 0      ] = val[ ( sty - 1     ) * n + stx - 1     ];
    sdata[ 0      ][ b2 - 1 ] = val[ ( sty - 1     ) * n + stx + BSIZE ];
    sdata[ b2 - 1 ][ 0      ] = val[ ( sty + BSIZE ) * n + stx - 1     ];
    sdata[ b2 - 1 ][ b2 - 1 ] = val[ ( sty + BSIZE ) * n + stx + BSIZE ];
  }
  if( ( j > 0 ) && ( tx == 1 ) ) {
    sdata[ ty     ][ 0      ] = val[ i * n + j - 1 ];
  }
  if( ( j < ( n - 1 ) ) && ( tx == BSIZE ) ) {
    sdata[ ty     ][ b2 - 1 ] = val[ i * n + j + 1 ];
  }
  if( ( i > 0 ) && ( ty == 1 ) ) {
    sdata[ 0      ][ tx     ] = val[ ( i - 1 ) * n + j ];
  }
  if( ( i < ( n - 1 ) ) && ( ty == BSIZE ) ) {
    sdata[ b2 - 1 ][ tx     ] = val[ ( i + 1 ) * n + j ];
  }
  __syncthreads( );
  if( ( i > 0 ) && ( i < ( n - 1 ) ) && ( j > 0 ) && ( j < ( n - 1 ) ) ) {
    estado = sdata[ ty ][ tx ];
    up = sdata[ ty - 1 ][ tx ];
    upright = sdata[ ty - 1 ][ tx + 1 ];
    right = sdata[ ty ][ tx + 1 ];
    rightdown = sdata[ ty + 1 ][ tx + 1 ];
    down = sdata[ ty + 1 ][ tx ];
    downleft = sdata[ ty + 1 ][ tx - 1 ];
    left = sdata[ ty ][ tx - 1 ];
    leftup = sdata[ ty - 1 ][ tx - 1 ];
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
  if( argc > 1 ) {
    n = atoi( argv[ 1 ] );
  }
  size = n * n;
  if( argc < 3 ) {
    steps = 4 * ( n - 5 );
    // steps = 4;
  }
  else {
    steps = atoi( argv[ 2 ] );
  }
  dim3 dimBlock( BSIZE, BSIZE );
  dim3 dimGrid( ceil( float( n ) / float( dimBlock.x ) ), ceil( float( n ) / float( dimBlock.y ) ) );
  printf("dimGrid = (%d x %d) \n", dimGrid.x, dimGrid.y);
  int *data_h = ( int* ) malloc( size * sizeof( int ) );
  int *data_d, *aux_d;
  cudaMalloc( ( void** ) &data_d, size * sizeof( int ) );
  cudaMalloc( ( void** ) &aux_d, size * sizeof( int ) );
  cudaMemset( aux_d, 0, size * sizeof( int ) );

  int i;
  for( i = 0; i < size; ++i ) {
    data_h[ i ] = 0;
  }
  data_h[ 1 * n + 2 ] = 1;
  data_h[ 2 * n + 3 ] = 1;
  data_h[ 3 * n + 1 ] = 1;
  data_h[ 3 * n + 2 ] = 1;
  data_h[ 3 * n + 3 ] = 1;

  print( data_h, n, 0, 12 );

  double start = omp_get_wtime( );
  cudaMemcpy( data_d, data_h, size * sizeof( int ), cudaMemcpyHostToDevice );
  int *aux2;
  for( int step = 0; step < steps; ++step ) {
    Evolve <<< dimGrid, dimBlock >>> ( data_d, aux_d, n );
    aux2 = data_d; data_d = aux_d; aux_d = aux2;
    // cudaMemcpy( data_h, data_d, size * sizeof( int ), cudaMemcpyDeviceToHost );
    // print( data_h, n, 0, n );
  }
  cudaMemcpy( data_h, data_d, size * sizeof( int ), cudaMemcpyDeviceToHost );
  print( data_h, n, n - 12, n );
  printf( "\tTempo total : \t %f \n", omp_get_wtime( ) - start );

  return( 0 );
}
