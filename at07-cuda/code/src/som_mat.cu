#include <cuda.h>
#include <stdio.h>

__global__ void MatrixAdd_d( float *A, float *B, float *C, int N ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = i * N + j;
  if( ( i < N ) && ( j < N ) ) {
    C[ index ] = A[ index ] + B[ index ];
  }
}

int main( ) {
  float *a_h, *b_h, *c_h; /* pointers to host memory; a.k.a. CPU */
  float *a_d, *b_d, *c_d; /* pointers to device memory; a.k.a. GPU */
  int blocksize = 16, n = 2, i, j, index;
  /* allocate arrays on host */
  a_h = ( float* ) malloc( sizeof( float ) * n * n );
  b_h = ( float* ) malloc( sizeof( float ) * n * n );
  c_h = ( float* ) malloc( sizeof( float ) * n * n );
  /* allocate arrays on device */
  cudaMalloc( ( void** ) &a_d, n * n * sizeof( float ) );
  cudaMalloc( ( void** ) &b_d, n * n * sizeof( float ) );
  cudaMalloc( ( void** ) &c_d, n * n * sizeof( float ) );
  dim3 dimBlock( blocksize, blocksize );
  dim3 dimGrid( ceil( float( n ) / float( dimBlock.x ) ), ceil( float( n ) / float( dimBlock.y ) ) );
  /* initialize the arrays */
  for( j = 0; j < n; j++ ) {
    for( i = 0; i < n; i++ ) {
      index = i * n + j;
      a_h[ index ] = rand( ) % 35;
      b_h[ index ] = rand( ) % 35;
    }
  }
  /* copy and run the code on the device */
  cudaMemcpy( a_d, a_h, n * n * sizeof( float ), cudaMemcpyHostToDevice );
  cudaMemcpy( b_d, b_h, n * n * sizeof( float ), cudaMemcpyHostToDevice );
  MatrixAdd_d << < dimGrid, dimBlock >> > ( a_d, b_d, c_d, n );
  cudaThreadSynchronize( );
  cudaMemcpy( c_h, c_d, n * n * sizeof( float ), cudaMemcpyDeviceToHost );
  /* print out the answer */
  for( j = 0; j < n; j++ ) {
    for( i = 0; i < n; i++ ) {
      index = i * n + j;
      /* This time the array is only 2x2 so we can print it out. */
      printf( "A + B = C: %d %d %f + %f = %f\n", i, j, a_h[ index ], b_h[ index ], c_h[ index ] );
    }
  }
  /* cleanup... */
  free( a_h );
  free( b_h );
  free( c_h );
  cudaFree( a_d );
  cudaFree( b_d );
  cudaFree( c_d );
  return( 0 );
}
