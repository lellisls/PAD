#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define tam 1.0
#define dx 0.00001
#define dt 0.000001
#define T 0.01
#define kappa 0.000045

__global__ void Inicializacao( double *uprev, const int n ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double x = idx * dx;
  if( idx < n + 1 ) {
    if( x <= 0.5 ) {
      uprev[ idx ] = 200 * x;
    }
    else {
      uprev[ idx ] = 200 * ( 1. - x );
    }
  }
}

__global__ void Atualiza( double *u, double *u_prev, const int n ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if( idx == 0 ) {
    u[ 0 ] = u[ n ] = 0.; /* forca condicao de contorno */
  }
  else if( idx < n ) {
    u[ idx ] = u_prev[ idx ] + kappa * dt / ( dx * dx ) * ( u_prev[ idx - 1 ] - 2 * u_prev[ idx ] + u_prev[ idx + 1 ] );
  }
}


__global__ void Maximo( double *input, double *results, int n ) {
  extern __shared__ double sdata[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x, tx = threadIdx.x;
  double x = 0.;
  if( idx < n + 1 ) {
    x = input[ idx ];
  }
  sdata[ tx ] = x;
  __syncthreads( );
  for( int offset = blockDim.x / 2; offset > 0; offset >>= 1 ) {
    if( tx < offset ) {
      if( sdata[ tx ] < sdata[ tx + offset ] ) {
        sdata[ tx ] = sdata[ tx + offset ];
      }
    }
  }
  __syncthreads( );
  if( threadIdx.x == 0 ) {
    results[ blockIdx.x ] = sdata[ 0 ];
  }
}

int main( void ) {

  double *tmp;
  double *u_prev_d, *u_d, *max_d;
  double t;
  long int n;
  /* Claculando quantidade de pontos */
  n = tam / dx;

  int blockSize = 256;
  int numBlocks = ceil( n / ( float ) blockSize );
  printf( "Size: %d, numBlks: %d, numThds: %d, mult: %d\n", n, numBlocks, blockSize, numBlocks * blockSize );

  cudaMalloc( ( void** ) &u_d, ( n + 1 ) * sizeof( double ) );
  cudaMalloc( ( void** ) &u_prev_d, ( n + 1 ) * sizeof( double ) );
  cudaMalloc( ( void** ) &max_d, ( n + 2 ) * sizeof( double ) );

  printf( "Inicio: qtde=%ld, dt=%g, dx=%g, dx²=%g, kappa=%f, const=%f\n",
          ( n + 1 ), dt, dx, dx * dx, kappa, kappa * dt / ( dx * dx ) );
  printf( "Iteracoes previstas: %g\n", T / dt );

  double start = omp_get_wtime( );

  Inicializacao << < numBlocks, blockSize >> > ( u_prev_d, n );

  /* cudaMemcpy( u_prev, u_prev_d, ( n + 1 ) * sizeof( double ), cudaMemcpyDeviceToHost ); */

  /*
   * for( i = 0; i < n + 1; i++ ) {
   *   printf( "%.2f ", u_prev[ i ] );
   * }
   * printf( "\n" );
   */
  /*
   * x = ( n + 1 ) * dx;
   * printf( "%f\n", x );
   * printf( "dx=%g, x=%g, x-dx=%g\n", dx, x, x - dx );
   * printf( "u_prev[0,1]=%g, %g\n", u_prev[ 0 ], u_prev[ 1 ] );
   * printf( "u_prev[n-1,n]=%g, %g\n", u_prev[ n - 1 ], u_prev[ n ] );
   */

  t = 0.;
  while( t < T ) {
    Atualiza << < numBlocks, blockSize >> > ( u_d, u_prev_d, n );
    tmp = u_prev_d; u_prev_d = u_d; u_d = tmp; /* troca entre ponteiros */
    t += dt;
  }
  int smem_sz = blockSize * sizeof( double );
  Maximo << < numBlocks, blockSize, smem_sz >> > ( u_d, max_d, n );
  Maximo << < 1, blockSize, smem_sz >> > ( max_d, max_d + numBlocks, numBlocks );

  double maxval;
  cudaMemcpy( &maxval, max_d + numBlocks, sizeof( double ), cudaMemcpyDeviceToHost );
  printf( "Tempo: \t %f \n", omp_get_wtime( ) - start );

  printf( "Maior valor = %g\n", maxval );

}
