#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "stats.h"

void printMatrix( float *mat, int width ) {
  int i, j;
  printf( "\n" );
  for( i = 0; i < width; ++i ) {
    const int isz = i * width;
    for( j = 0; j < width; ++j ) {
      printf( "%08.2f ", mat[ isz + j ] );
    }
    printf( "\n" );
  }
}

int main( int argc, char **argv ) {
  int num_thds = 1;
  int width = 1024;
  int block = 16;
  char str[ 20 ];
  if( argc == 4 ) {
    width = atoi( argv[ 1 ] );
    block = atoi( argv[ 2 ] );
    num_thds = atoi( argv[ 3 ] );
  }
  else {
    printf( "usage: <width> <block> <num_thds>\n" );
    return( 0 );
  }
  /* INICIALIZAÇÃO */
  PRINT( printf( "Inicializando Matriz: %dx%d\n", width, width ) );
  PRINT( printf( "Tamanho do bloco: %d\n", block ) );
  int i, j, k;
  int ii, jj, kk;
  float *a, *b, *c;
  const int size = width * width;

  a = ( float* ) malloc( size * sizeof( float ) );
  b = ( float* ) malloc( size * sizeof( float ) );
  c = ( float* ) calloc( size, sizeof( float ) );
  srand( 424242 );
  for( i = 0; i < size; ++i ) {
    a[ i ] = ( rand( ) % 10000 ) / 100.0f;
    b[ i ] = ( rand( ) % 10000 ) / 100.0f;
  }
  PRINT( printMatrix( a, width ); );
  PRINT( printMatrix( b, width ); );

  omp_set_num_threads( num_thds );

  inicializacao( );
  /* FUNÇÃO A SER AVALIADA */
#pragma omp parallel for default(none) shared(a, b, c) private(ii, jj, kk, i, j, k) firstprivate(block, width)
  for( ii = 0; ii < width; ii += block ) {
    for( kk = 0; kk < width; kk += block ) {
      for( jj = 0; jj < width; jj += block ) {
        for( i = ii; i < ii + block; i++ ) {
          const int isz = i * width;
          for( k = kk; k < kk + block; k++ ) {
            const int ksz = k * width;
            for( j = jj; j < jj + block; j++ ) {
              c[ isz + j ] = c[ isz + j ] + a[ isz + k ] * b[ ksz + j ];
            }
          }
        }
      }
    }
  }

  sprintf( str, "%d; %d; %d", width, block, num_thds );
  avaliacao( str, size );
  PRINT( printMatrix( c, width ); );

  volatile float ajsaks = c[ 0 ];
  /* free( a ); free( b ); free( c ); */
  return( 0 );
}
