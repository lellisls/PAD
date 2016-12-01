#include <cuda.h>
#include <omp.h>

// #include "stats.h"
#include "stdio.h"
#include "stdlib.h"
#define SIZE 1000
__global__ void Evolve( int *val, int *aux, int n ) {
  int up, upright, right, rightdown, down, downleft, left, leftup;
  int sum = 0, estado;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if ( i > 0 && i < (n - 1) && j > 0 && j < (n - 1) ){
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
    // steps = 4;
  }
  else {
    steps = atoi( argv[ 2 ] );
  }

  int blocksize = 16;
  dim3 dimBlock( blocksize, blocksize );
  dim3 dimGrid( ceil( float( n ) / float( dimBlock.x ) ), ceil( float( n ) / float( dimBlock.y ) ) );

  int *data_h = ( int* ) malloc( size * sizeof( int ) );
  int *data_d, *aux_d;
  cudaMalloc( ( void** ) &data_d, size * sizeof( int ) );
  cudaMalloc( ( void** ) &aux_d, size * sizeof( int ) );
  cudaMemset( aux_d, 0, size*sizeof(int));

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

  double start = omp_get_wtime();
  cudaMemcpy(data_d, data_h, size * sizeof( int ), cudaMemcpyHostToDevice);
  int *aux2;
  for( int step = 0; step < steps; ++step ) {
    Evolve<<< dimGrid, dimBlock >>>( data_d, aux_d, n );
    aux2 = data_d; data_d = aux_d; aux_d = aux2;
    // cudaMemcpy(data_h, data_d, size * sizeof( int ), cudaMemcpyDeviceToHost);
    // print( data_h, n, n - 12, n );
  }
  cudaMemcpy(data_h, data_d, size * sizeof( int ), cudaMemcpyDeviceToHost);
  print( data_h, n, n - 12, n );
  printf( "\tTempo total : \t %f \n", omp_get_wtime( ) - start );

  return( 0 );
}
