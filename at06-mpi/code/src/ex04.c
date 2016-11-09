#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define RGB_COMPONENT_COLOR 255

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
  fp = stdin;
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

  printf( "%d: %d\n", rank, n );
  cols = image->x;
  rows = image->y;
  for( i = 0; i < n; i++ ) {
    image->data[ i ].red = floor( ( image->data[ i ].red * 4 ) / 256 );
    image->data[ i ].blue = floor( ( image->data[ i ].blue * 4 ) / 256 );
    image->data[ i ].green = floor( ( image->data[ i ].green * 4 ) / 256 );
  }
  count = 0;
  x = 0;
  for( j = 0; j <= 3; j++ ) {
    for( k = 0; k <= 3; k++ ) {
      for( l = 0; l <= 3; l++ ) {
#pragma omp parallel for firstprivate(j, k, l) shared(image) reduction(+:count)
        for( i = 0; i < n; ++i ) {
          if( ( image->data[ i ].red == j ) && ( image->data[ i ].green == k ) && ( image->data[ i ].blue == l ) ) {
            count++;
          }
        }
        h[ x ] = ( float ) count / globalSize;
        count = 0;
        x++;
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
  MPI_Datatype pixelType, oldtypes;
  int blockcounts;
  MPI_Aint offsets, extent;

  offsets = 0;
  oldtypes = MPI_INT;
  blockcounts = 3;

  localImage = (PPMImage*) malloc(sizeof(PPMImage));

  MPI_Type_create_struct( 1, &blockcounts, &offsets, &oldtypes, &pixelType );
  MPI_Type_commit( &pixelType );
  if( rank == 0 ) {
    image = readPPM( );
    xsz = image->x;
    ysz = image->y;
    n = xsz * ysz;
    printf("%d x %d\n", xsz, ysz);
    for( iRank = 0; iRank < commSize; ++iRank ) {
      int rows = floor( ( float ) ysz / commSize );
      int first = iRank * rows * xsz;
      if( iRank == commSize - 1 ) {
        rows = ysz - iRank * rows;
      }
      int pixels = rows * xsz;
      int last = first + pixels;
      if( iRank == 0 ){
        localImage->x = xsz;
        localImage->y = rows;
        localImage->data = image->data;
      }else{
        int localSize[2] = { xsz, rows};
        printf( "%d: %d x %d, %d pixels --- %d -> %d\n", iRank, xsz, rows, pixels, first, last );
        MPI_Send( &n, 1, MPI_INT, iRank, 100, MPI_COMM_WORLD );
        MPI_Send( &localSize[ 0 ], 2, MPI_INT, iRank, 101, MPI_COMM_WORLD );
        if(rows != 0){
          MPI_Send( &image->data[ first ], pixels, pixelType, iRank, 102, MPI_COMM_WORLD );
        }
      }
    }
  }
  else {
    MPI_Recv( &n, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    MPI_Recv( &localImage->x, 2, MPI_INT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    xsz = localImage->x; ysz = localImage->y;
    localImage->data = ( PPMPixel* ) malloc( xsz * ysz * sizeof( PPMPixel ) );
    if(ysz != 0){
      printf("Local image size: %d X %d\n", xsz, ysz);
      MPI_Recv( &localImage->data[ 0 ], xsz * ysz , pixelType, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
  }
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
  free( h );
  free( res );

  MPI_Finalize( );
}
