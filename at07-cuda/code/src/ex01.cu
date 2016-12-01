#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define tam 1.0
#define dx 0.00001
#define dt 0.000001
#define T 0.01
#define kappa 0.000045

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
  if( idx < n ) {
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
    __syncthreads( );
  }
  if( threadIdx.x == 0 ) {
    results[ blockIdx.x ] = sdata[ 0 ];
  }
}

int main( void ) {

  double *tmp;
  double *u_prev;
  double *u_prev_d, *u_d, *max_d;
  double t;
  long int n;
  /* Claculando quantidade de pontos */
  n = tam / dx;

  int blockSize = 256;
  int gridSize = ceil( n / ( float ) blockSize );
  printf( "Size: %d, numBlks: %d, numThds: %d, mult: %d\n", n, gridSize, blockSize, gridSize * blockSize );

  u_prev = ( double* ) malloc( ( n + 1 ) * sizeof( double ) );

  cudaMalloc( ( void** ) &u_d, ( n + 1 ) * sizeof( double ) );
  cudaMalloc( ( void** ) &u_prev_d, ( n + 1 ) * sizeof( double ) );
  cudaMalloc( ( void** ) &max_d, (gridSize + 1 )* sizeof( float )  * sizeof( double ) );

  printf( "Inicio: qtde=%ld, dt=%g, dx=%g, dx²=%g, kappa=%g, const=%g\n",
          ( n + 1 ), dt, dx, dx * dx, kappa, kappa * dt / ( dx * dx ) );
  printf( "Iteracoes previstas: %g\n", T / dt );

  double start = omp_get_wtime( );

  double x = 0.;
  for( int i = 0; i < n + 1; i++ ) {
    if( x <= 0.5 ) {
      u_prev[ i ] = 200 * x;
    }
    else {
      u_prev[ i ] = 200 * ( 1. - x );
    }
    x += dx;
  }
  double start_copy1 = omp_get_wtime( );

  cudaMemcpy( u_prev_d, u_prev, sizeof( double ) * (n + 1), cudaMemcpyHostToDevice );

  printf( "\tHostToDevice : %g\n", omp_get_wtime( ) - start_copy1 );
  printf( "\tInicializacao: %g\n", omp_get_wtime( ) - start );

  /* cudaMemcpy( u_prev, u_prev_d, ( n + 1 ) * sizeof( double ), cudaMemcpyDeviceToHost ); */

  /*
   * for( i = 0; i < n + 1; i++ ) {
   *   printf( "%.2f ", u_prev[ i ] );
   * }
   * printf( "\n" );
   */
  /*
   * x = ( n + 1 ) * dx;
   * printf( "%g\n", x );
   * printf( "dx=%g, x=%g, x-dx=%g\n", dx, x, x - dx );
   * printf( "u_prev[0,1]=%g, %g\n", u_prev[ 0 ], u_prev[ 1 ] );
   * printf( "u_prev[n-1,n]=%g, %g\n", u_prev[ n - 1 ], u_prev[ n ] );
   */

  t = 0.;

  double start_kernel = omp_get_wtime( );
  int steps = 0;
  while( t < T ) {
    Atualiza << < gridSize, blockSize >> > ( u_d, u_prev_d, n );
    tmp = u_prev_d; u_prev_d = u_d; u_d = tmp; /* troca entre ponteiros */
    t += dt;
    ++steps;
  }
  double totalKernel = omp_get_wtime( ) - start_kernel;
  printf( "\tTempo kernels: %g \n", totalKernel );
  printf( "\tTempo medio  : %g \n", totalKernel / steps );

  int smem_sz = blockSize * sizeof( double );
  Maximo << < gridSize, blockSize, smem_sz >> > ( u_d, max_d, n );
  Maximo << < 1, blockSize, smem_sz >> > ( max_d, max_d + gridSize, gridSize );

  double maxval;
  double start_copy2 = omp_get_wtime( );
  cudaMemcpy( &maxval, max_d + gridSize, sizeof( double ), cudaMemcpyDeviceToHost );
  printf( "\tDeviceToHost : %g \n", omp_get_wtime( ) - start_copy2 );
  printf( "\tTempo Total  : %g \n", omp_get_wtime( ) - start );

  printf( "Maior valor = %g\n", maxval );

}
