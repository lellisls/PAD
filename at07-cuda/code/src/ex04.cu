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
  // results[ idx ] = curand(&state) / 1000.0f;
  if( idx < n ){
    results[ idx ] = 1.0;
  }
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

int main( void ) {

  float *data_d, *result_d;
  long int n = 100000;

  int blockSize = 256;
  int gridSize = ceil( n / ( float ) blockSize );
  int realsize = gridSize * blockSize;
  printf( "Size: %d, numBlks: %d, numThds: %d, mult: %d\n", n, gridSize, blockSize, realsize);

  cudaMalloc( ( void** ) &data_d, realsize * sizeof( float ) );
  cudaMalloc( ( void** ) &result_d, (gridSize + 1 ) * sizeof( float )  );
  double start = omp_get_wtime();

  Random<<< gridSize, blockSize >>> ( data_d, n, time(NULL));

  int smem_sz = blockSize * sizeof( float );
  Somatorio <<< gridSize, blockSize, smem_sz >>> ( data_d, result_d, n );
  Somatorio <<< 1, blockSize, smem_sz >>> ( result_d, result_d + gridSize, gridSize );


  float sum;
  double start_copy = omp_get_wtime( );
  cudaMemcpy( &sum, result_d + gridSize , sizeof( float ), cudaMemcpyDeviceToHost );
  printf( "\tTempo transf : %f \n", omp_get_wtime( ) - start_copy );
  printf( "\tTempo Total  : %f \n", omp_get_wtime( ) - start );

  printf( "Somatorio = %g\n", sum );


}
