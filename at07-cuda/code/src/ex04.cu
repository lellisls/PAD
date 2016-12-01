#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

__global__ void Random( float *results, long int n, unsigned int seed ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;

  curand_init(seed, blockIdx.x, 0, &state);
  results[ idx ] = curand(&state) / 1000.0f;
  // if( idx < n ){
  //   results[ idx ] = 1.0;
  // }
}

__global__ void Somatorio( float *input, float *results, long int n ) {
  extern __shared__ float sdata[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x, tx = threadIdx.x;
  float x = 0.;
  if( idx < n ) {
    x = input[ idx ];
  }
  sdata[ tx ] = x;
  __syncthreads( );
  for( int offset = blockDim.x / 2; offset > 0; offset >>= 1 ) {
    if( tx < offset ) {
      sdata[ tx ] += sdata[ tx + offset ];
    }
    __syncthreads( );
  }
  if( threadIdx.x == 0 ) {
    results[ blockIdx.x ] = sdata[ 0 ];
  }
}

float sum(float *data_d, size_t n) {
  int blockSize = 1024, gridSize = ceil( n / ( float ) blockSize );
  float * sums_d;

  printf( "Size: %d, numBlks: %d, numThds: %d, mult: %d\n", n, gridSize, blockSize, blockSize * gridSize );

  cudaMalloc( ( void** ) &sums_d, (gridSize + 1 ) * sizeof( float )  );

  int smem_sz = blockSize * sizeof( float );
  Somatorio <<< gridSize, blockSize, smem_sz >>> ( data_d, sums_d, n );
  Somatorio <<< 1, blockSize, smem_sz >>> ( sums_d, sums_d, gridSize );
  float totalf;
  cudaMemcpy( &totalf, sums_d, sizeof( float ), cudaMemcpyDeviceToHost );
  return totalf;
}

int main( void ) {

  float *data_d;
  long int n = 1000000;

  int blockSize = 1024;
  int gridSize = ceil( n / ( float ) blockSize );
  int realsize = gridSize * blockSize;
  printf( "Size: %d, numBlks: %d, numThds: %d, mult: %d\n", n, gridSize, blockSize, realsize);

  cudaMalloc( ( void** ) &data_d, realsize * sizeof( float ) );

  double start = omp_get_wtime();

  Random<<< gridSize, blockSize >>> ( data_d, n, time(NULL));

  printf( "\tTempo Total  : %f \n", omp_get_wtime( ) - start );

  printf( "Somatorio = %f\n", sum(data_d, n) );
}

//Citar no relatório que o tamanho do bloco pode levar a um resultado errado,
// quando o numero de blocos é menor que o nro de threads.
