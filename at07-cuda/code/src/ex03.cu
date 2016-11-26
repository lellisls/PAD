#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

__global__ void Random( float *results, int n, unsigned int seed ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t state;

  curand_init(seed, blockIdx.x, 0, &state);
  results[ idx ] = curand(&state) / 1000.0f;
}

int main( void ) {

  float *data_d;
  long int n = 1000000;

  int blockSize = 256;
  int numBlocks = ceil( n / ( float ) blockSize );
  int realsize = numBlocks * blockSize;
  printf( "Size: %d, numBlks: %d, numThds: %d, mult: %d\n", n, numBlocks, blockSize, realsize);

  cudaMalloc( ( void** ) &data_d, realsize * sizeof( float ) );
  double start = omp_get_wtime();

  Random<<< numBlocks, blockSize >>> ( data_d, n, time(NULL));

  printf( "\tTempo Total  : %f \n", omp_get_wtime( ) - start );

}
