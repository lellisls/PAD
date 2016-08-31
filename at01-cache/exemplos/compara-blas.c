#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define REP 500

/* compilar: gcc fonte.c -o exec -lblas */

int main( void ) {
  float *x, *y, *y2;
  int i, j;
  double ti = 0, tf = 0; /* ti = tempo inicial // tf = tempo final */
  struct timeval tempo_inicio, tempo_fim;

  /* aloca vetores dianmicamente */

  x = ( float* ) malloc( N * sizeof( float ) );
  y = ( float* ) malloc( N * sizeof( float ) );
  y2 = ( float* ) malloc( N * sizeof( float ) );
  /* atribue valores aleatorios */
  for( i = 0; i < N; i++ ) {
    x[ i ] = random( );
    y2[ i ] = y[ i ] = random( );
  }
  /* marca tempo inicial */
  gettimeofday( &tempo_inicio, NULL );
  /* Calcula y=alfa*x+y */
  for( j = 0; j < REP; j++ ) {
    cblas_saxpy( N, 0.5, x, 1, y, 1 );
  }
  /* calcula o tempo gasto */
  gettimeofday( &tempo_fim, NULL );
  tf = ( double ) tempo_fim.tv_usec / 1000000.0 + tempo_fim.tv_sec;
  ti = ( double ) tempo_inicio.tv_usec / 1000000.0 + tempo_inicio.tv_sec;
  printf( "Tempo gasto em saxpy (BLAS) %.15f segundos \n", tf - ti );

  /* marca tempo inicial */
  gettimeofday( &tempo_inicio, NULL );
  /* Calcula y=alfa*x+y */
  for( j = 0; j < REP; j++ ) {
    for( i = 0; i < N; i++ ) {
      y2[ i ] = 0.5 * x[ i ] + y2[ i ];
    }
  }
  /* calcula o tempo gasto */
  gettimeofday( &tempo_fim, NULL );
  tf = ( double ) tempo_fim.tv_usec / 1000000.0 + tempo_fim.tv_sec;
  ti = ( double ) tempo_inicio.tv_usec / 1000000.0 + tempo_inicio.tv_sec;
  printf( "Tempo gasto direto %.15f segundos \n", tf - ti );

  return( 0 );
}
