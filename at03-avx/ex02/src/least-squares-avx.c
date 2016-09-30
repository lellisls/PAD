/* ------------------------------------------------------------------------
* FILE: least-squares.c
* This program computes a linear model for a set of given data.
*
* PROBLEM DESCRIPTION:
*  The method of least squares is a standard technique used to find
*  the equation of a straight line from a set of data. Equation for a
*  straight line is given by
*	 y = mx + b
*  where m is the slope of the line and b is the y-intercept.
*
*  Given a set of n points {(x1,y1), x2,y2),...,xn,yn)}, let
*      SUMx = x1 + x2 + ... + xn
*      SUMy = y1 + y2 + ... + yn
*      SUMxy = x1*y1 + x2*y2 + ... + xn*yn
*      SUMxx = x1*x1 + x2*x2 + ... + xn*xn
*
*  The slope and y-intercept for the least-squares line can be
*  calculated using the following equations:
*        slope (m) = ( SUMx*SUMy - n*SUMxy ) / ( SUMx*SUMx - n*SUMxx )
*  y-intercept (b) = ( SUMy - slope*SUMx ) / n
*
* AUTHOR: Dora Abdullah (Fortran version, 11/96)
* REVISED: RYL (converted to C, 12/11/96)
* ---------------------------------------------------------------------- */
#include "stats.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h> /* where intrinsics are defined */
#include <xmmintrin.h>

int main( int argc, char **argv ) {

  double *x, *y;
  double SUMx, SUMy, SUMxy, SUMxx, SUMres, res, slope,
         y_intercept, y_estimate;
  __m256d data_x, data_y, acc_x, acc_y, acc_xy, acc_xx, mult_xx, mult_xy;
  int i, n;
  FILE *infile;
  if( argc > 1 ) {
    infile = fopen( argv[ 1 ], "r" );
    if( infile == NULL ) {
      printf( "error opening file '%s'\n", argv[ 1 ] );
    }
    fscanf( infile, "%d", &n );
    x = ( double* ) _mm_malloc( n * sizeof( double ), 32 );
    y = ( double* ) _mm_malloc( n * sizeof( double ), 32 );
    for( i = 0; i < n; i++ ) {
      fscanf( infile, "%lf %lf", &x[ i ], &y[ i ] );
    }
  }
  else {
    n = 100000000;
    srand( 424242 );
    x = ( double* ) _mm_malloc( n * sizeof( double ), 32 );
    y = ( double* ) _mm_malloc( n * sizeof( double ), 32 );
    for( i = 0; i < n; i++ ) {
      x[ i ] = i * 2 + rand( ) / 100000000.0;
      y[ i ] = i * 5 - rand( ) / 10000000.0;
    }
  }
  inicializacao( );
  for( i = 0; i < n - 4; i += 4 ) {
    data_x = _mm256_load_pd( x + i );
    acc_x = _mm256_add_pd( acc_x, data_x );
    // SUMx += x[ i ];
    data_y = _mm256_load_pd( y + i );
    acc_y = _mm256_add_pd( acc_y, data_y );
    // SUMy += y[ i ];
    mult_xy = _mm256_mul_pd( data_x, data_y );
    acc_xy = _mm256_add_pd( acc_xy, mult_xy );
    // SUMxy += x[ i ] * y[ i ];
    mult_xx = _mm256_mul_pd( data_x, data_x );
    acc_xx = _mm256_add_pd( acc_xx, mult_xx );
    // SUMxx += x[ i ] * x[ i ];
  }
  SUMx = 0; SUMy = 0; SUMxy = 0; SUMxx = 0;
  for( ; i < n; i++ ) {
    SUMx += x[ i ];
    SUMy += y[ i ];
    SUMxy += x[ i ] * y[ i ];
    SUMxx += x[ i ] * x[ i ];
  }
  for( i = 0; i < 4; i++ ) {
    SUMx += acc_x[ i ];
    SUMy += acc_y[ i ];
    SUMxy += acc_xy[ i ];
    SUMxx += acc_xx[ i ];
  }
  slope = ( SUMx * SUMy - n * SUMxy ) / ( SUMx * SUMx - n * SUMxx );
  y_intercept = ( SUMy - slope * SUMx ) / n;
  avaliacao( "Least Squares - AVX", n );

  PRINT( printf( "\n" ); );
  PRINT( printf( "The linear equation that best fits the given data:\n" ); );
  PRINT( printf( "       y = %6.2lfx + %6.2lf\n", slope, y_intercept ); );
  PRINT( printf( "--------------------------------------------------\n" ); );
  PRINT( printf( "   Original (x,y)     Estimated y     Residual\n" ); );
  PRINT( printf( "--------------------------------------------------\n" ); );
  volatile double result = y_intercept;
  /*
   * SUMres = 0;
   * for( i = 0; i < n; i++ ) {
   *   y_estimate = slope * x[ i ] + y_intercept;
   *   res = y[ i ] - y_estimate;
   *   SUMres = SUMres + res * res;
   *   printf( "   (%6.2lf %6.2lf)      %6.2lf       %6.2lf\n",
   *           x[ i ], y[ i ], y_estimate, res );
   * }
   * printf( "--------------------------------------------------\n" );
   * printf( "Residual sum = %6.2lf\n", SUMres );
   */
}
