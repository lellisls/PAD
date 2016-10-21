#include <stdio.h>
#include <stdlib.h>
#include "stats.h"

#define nmax 32000
/* 32000 */



void show_usage( );
unsigned long* create_sieve_to_number( unsigned long number );

int main( int argc, char const *argv[] ) {

  unsigned long number;
  int num_thds = 4;
  unsigned long *sieve;
  if( argc == 2 ){
    num_thds = atoi(argv[1]);
  }
  int *qtty;
  qtty = ( int* ) malloc( nmax * sizeof( int ) );

  inicializacao();

  sieve = create_sieve_to_number( nmax );
  // static, 10
  // static, 5
  // static, 2
  // dynamic
  // guided
  #pragma omp parallel for schedule(SCHEDULE)
  for( number = 2; number < nmax; number += 2 ) {
    for( unsigned long i = 2; i < number; i++ ) {
      if( sieve[ i ] == 1 ) {
        for( unsigned long j = i; j < number; j++ ) {
          if( sieve[ j ] == 1 ) {
            if( i + j == number ) {
              qtty[ number ]++;
              // printf("Solution found: %ld + %ld = %ld\n", i, j, number);
              break;
            }
          }
        }
        if( qtty[ number ] > 0 ) {
          break;
        }
      }
    }
  }

  avaliacao(SCHEDULE_TXT, 3200);
  /* printf("no solution found!  pick up your Fields Medal!\n"); */


  return( 0 );

}

void show_usage( void ) {
  printf( "usage: goldbach [even number]\n" );
}

unsigned long* create_sieve_to_number( unsigned long number ) {
  unsigned long *sieve;

  sieve = ( unsigned long* ) malloc( sizeof( unsigned long ) * ( number + 1 ) );
  for( int i = 0; i < number; i++ ) {
    sieve[ i ] = 1;
  }
  for( unsigned long i = 2; i < number; i++ ) {
    if( sieve[ i ] == 1 ) {
      for( unsigned long j = i * i; j < number; j = j + i ) {
        sieve[ j ] = 0;
      }
    }
  }
  return( sieve );
}
