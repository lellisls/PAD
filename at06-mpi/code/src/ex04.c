#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define RGB_COMPONENT_COLOR 255

#ifndef NUMTHDS
#define NUMTHDS 1
#endif

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage* readPPM( ) {
  char buff[ 16 ];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen("large-image.ppm","r");
  if( !fgets( buff, sizeof( buff ), fp ) ) {
    perror( "stdin" );
    exit( 1 );
  }
  if( ( buff[ 0 ] != 'P' ) || ( buff[ 1 ] != '6' ) ) {
    fprintf( stderr, "Invalid image format (must be 'P6')\n" );
    exit( 1 );
  }
  img = ( PPMImage* ) malloc( sizeof( PPMImage ) );
  if( !img ) {
    fprintf( stderr, "Unable to allocate memory\n" );
    exit( 1 );
  }
  c = getc( fp );
  while( c == '#' ) {
    while( getc( fp ) != '\n' ) {
    }
    c = getc( fp );
  }
  ungetc( c, fp );
  if( fscanf( fp, "%d %d", &img->x, &img->y ) != 2 ) {
    fprintf( stderr, "Invalid image size (error loading)\n" );
    exit( 1 );
  }
  if( fscanf( fp, "%d", &rgb_comp_color ) != 1 ) {
    fprintf( stderr, "Invalid rgb component (error loading)\n" );
    exit( 1 );
  }
  if( rgb_comp_color != RGB_COMPONENT_COLOR ) {
    fprintf( stderr, "Image does not have 8-bits components\n" );
    exit( 1 );
  }
  while( fgetc( fp ) != '\n' ) {
  }
  img->data = ( PPMPixel* ) malloc( img->x * img->y * sizeof( PPMPixel ) );
  if( !img ) {
    fprintf( stderr, "Unable to allocate memory\n" );
    exit( 1 );
  }
  if( fread( img->data, 3 * img->x, img->y, fp ) != img->y ) {
    fprintf( stderr, "Error loading image.\n" );
    exit( 1 );
  }
  return( img );
}


void Histogram( PPMImage *image, float *h, int globalSize ) {
  int i, j, k, l, x, count;
  int rows, cols, rank;
  int n = image->y * image->x;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  printf( "%d: %dx%d\n", rank, image->x, image->y );
  cols = image->x;
  rows = image->y;
  #pragma omp parallel for
  for( i = 0; i < n; i++ ) {
    image->data[ i ].red = floor( ( image->data[ i ].red * 4 ) / 256 );
    image->data[ i ].blue = floor( ( image->data[ i ].blue * 4 ) / 256 );
    image->data[ i ].green = floor( ( image->data[ i ].green * 4 ) / 256 );
  }
  count = 0;
  x = 0;
  #pragma omp parallel for private(count, x, j, k, l) shared(image, globalSize, h)
  for( j = 0; j <= 3; j++ ) {
    for( k = 0; k <= 3; k++ ) {
      for( l = 0; l <= 3; l++ ) {
        x = j * 16 + k * 4 + l;
        count = 0;
        for( i = 0; i < n; ++i ) {
          if( ( image->data[ i ].red == j ) && ( image->data[ i ].green == k ) && ( image->data[ i ].blue == l ) ) {
            count++;
          }
        }
        h[ x ] = ( float ) count / globalSize;
        // x++;
      }
    }
  }
}

int main( int argc, char *argv[] ) {
  int rank, commSize, i, n, iRank, xsz, ysz;
  PPMImage *image, *localImage;

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &commSize );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Datatype pixelType;

  MPI_Request *requests = ( MPI_Request* ) malloc( sizeof( MPI_Request ) * commSize );

  omp_set_num_threads(NUMTHDS);

  MPI_Type_contiguous(3, MPI_CHAR, &pixelType);


  localImage = (PPMImage*) malloc(sizeof(PPMImage));
  MPI_Type_commit( &pixelType );
  if( rank == 0 ) {
    image = readPPM( );
    xsz = image->x;
    ysz = image->y;
    n = xsz * ysz;
    printf("Input image size: %d x %d\n", xsz, ysz);
    for( iRank = 0; iRank < commSize; ++iRank ) {
      int rows = floor( ( float ) ysz / commSize );
      int first = iRank * rows * xsz;
      if( iRank == commSize - 1 ) {
        rows = ysz - iRank * rows;
      }
      int pixels = rows * xsz;
      int last = first + pixels;
      printf( "Sending data to %d: %d x %d, %d pixels --- %d -> %d\n", iRank, xsz, rows, pixels, first, last );
      if( iRank == 0 ){
        localImage->x = xsz;
        localImage->y = rows;
        localImage->data = image->data;
      }else{
        int localSize[2] = { xsz, rows};
        // printf("Sending n to %d: %d\n", iRank, n);
        MPI_Isend( &n, 1, MPI_INT, iRank, 100, MPI_COMM_WORLD, &requests[iRank-1] );
        // printf("Sending localSize to %d: %dx%d\n", iRank, localSize[0], localSize[1]);
        MPI_Isend( &localSize[ 0 ], 2, MPI_INT, iRank, 101, MPI_COMM_WORLD, &requests[iRank-1] );
        // printf("Sending image to %d: %d pixels\n", iRank, pixels);
        PPMPixel * data = image->data;
        if(rows != 0){
          MPI_Isend( &data[ first ], pixels, pixelType, iRank, 102, MPI_COMM_WORLD, &requests[iRank-1] );
        }
      }
    }
  }
  else {
    MPI_Recv( &n, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    // printf("%d:  1st receive: %d\n", rank, n);
    int localSize[2];
    MPI_Recv( &localSize[0], 2, MPI_INT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    // printf("%d:  2nd receive: %dx%d\n", rank, localSize[0], localSize[1]);
    localImage->x = localSize[0];
    localImage->y = localSize[1];
    xsz = localImage->x; ysz = localImage->y;
    localImage->data = ( PPMPixel* ) malloc( xsz * ysz * sizeof( PPMPixel ) );
    PPMPixel * data = localImage->data;
    // printf("%d:  Expecting to receive %d pixels\n", rank, xsz * ysz);
    if(ysz != 0){
      MPI_Recv( &data[ 0 ], xsz * ysz , pixelType, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
    // printf("%d:  3rd receive: %d", rank, xsz * ysz);
  }
  // printf("%d:   Local image size: %d X %d\n", rank, localImage->x, localImage->y);

  float *h = ( float* ) malloc( sizeof( float ) * 64 );
  for( i = 0; i < 64; i++ ) {
    h[ i ] = 0.0;
  }
  if(localImage->y != 0){
    Histogram( localImage, h, n );
  }

  float *res = ( float* ) malloc( sizeof( float ) * 64 );
  MPI_Reduce(&h[0], &res[0], 64, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD );

  if(rank == 0){
    for( i = 0; i < 64; i++ ) {
      printf( "%0.3f ", res[ i ] );
    }
    printf( "\n" );
  }

  free(requests);
  if(rank != 0){
    free(localImage->data);
    free(localImage);
  }else{
    free(image->data);
    free(localImage);
    free(image);
  }
  free( h );
  free( res );

  MPI_Finalize( );
}
