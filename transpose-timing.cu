#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

typedef float real_t;

static const int TILE_DIM = 21; //initialized in main


__global__ void transpose( real_t *odata, real_t *idata, int width, int height) {
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int index_in = xIndex + yIndex * width;
	int index_out = yIndex + xIndex * height;
	odata[ index_out ] = idata[ index_in ];
}

/// Transpose matrix by first copying element into local cache.
/// Coalescing happens when tiles are multiple of 16
///# select indices of element to copy into shared memory
///# select target block indides: if input block is I,J target block is J, I
///# use threadIdx to select target element in target block
///# copy transposed element from local share; if elements are copied into local
///  share in [threadIdx.y][threadIdx.x] then target element must be copied from
///  [threadIdx.x][threadIdx.y]
__global__ void transposeCoalesced( real_t *odata, real_t *idata, int width, int height ) {
	__shared__ real_t tile[ TILE_DIM ][ TILE_DIM /*+1*/ ];
	const int xBlock = blockIdx.x * blockDim.x;
	const int yBlock = blockIdx.y * blockDim.y;
	int xIndex = xBlock + threadIdx.x;
	int yIndex = yBlock + threadIdx.y;
	const int index_in = xIndex + yIndex * width;
	xIndex = yBlock + threadIdx.x;
	yIndex = xBlock + threadIdx.y;
	const int index_out = xIndex + (yIndex)*height;
	tile[ threadIdx.y ][ threadIdx.x ] = idata[ index_in ];
	__syncthreads();
	odata[ index_out ] = tile[ threadIdx.x ][ threadIdx.y ];
}

__global__ void initMatrix( real_t* in ) {
	const int c = threadIdx.x + blockDim.x * blockIdx.x;
	const int r = threadIdx.y + blockDim.y * blockIdx.y;
	const int idx = c + gridDim.x * blockDim.x * r; 
	in[ idx ] = ( real_t ) idx; 
}

void printMatrix( const real_t* m, int r, int c ) {
	for( int i = 0; i != r; ++i ) {
		for( int j = 0; j != c; ++j )
			std::cout << m[ i * c + j ] << ' ';
		std::cout << '\n';
	}
	std::cout << std::endl;		
}

real_t strToReal( const char* str ) {
	if( !str ) throw std::runtime_error( "strToReal - NULL srting");
	std::istringstream is( str );
	real_t v = real_t();
	is >> v;
	return v;
}


int main(int argc, char** argv ) {
	
	const int DEFROWS = 30;
	const int DEFCOLUMNS = 40; 

	int ROWS = DEFROWS * TILE_DIM;
	int COLUMNS = DEFCOLUMNS * TILE_DIM;
	bool CSV = false;

	if( argc < 3 || argc > 4 ) {
		std::cout << "usage: " << argv[ 0 ] << " <# tile rows> <# tile columns> [csv]\n";
		std::cout << "  using default: tile size=" << TILE_DIM << " tile rows=" << ROWS << " tile columns=" << COLUMNS << std::endl;
	} else {
		ROWS = TILE_DIM * strToReal( argv[ 1 ] );
		COLUMNS = TILE_DIM * strToReal( argv[ 2 ] );
		if( argc == 4 ) { CSV = std::string( argv[ 3 ] ) == "csv"; }
	}

	const dim3 BLOCKS( COLUMNS / TILE_DIM, ROWS / TILE_DIM );
	const dim3 THREADS_PER_BLOCK( TILE_DIM, TILE_DIM );
	const size_t SIZE = ROWS * COLUMNS * sizeof( real_t );
	real_t* dev_in = 0;
	real_t* dev_out = 0;
	std::vector< real_t > outmatrix( ROWS * COLUMNS, 0.f );

	cudaMalloc( &dev_in,  SIZE );
	cudaMalloc( &dev_out, SIZE );

	initMatrix<<<dim3( COLUMNS, ROWS ), 1>>>( dev_in );
	cudaMemcpy( &outmatrix[ 0 ], dev_in, SIZE, cudaMemcpyDeviceToHost );

	//std::cout << "INPUT MATRIX - " << ROWS << " rows, " << COLUMNS << " columns" << std::endl;
	//printMatrix( &outmatrix[ 0 ], ROWS, COLUMNS );

	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop  = cudaEvent_t(); 
	float elapsed = 0.f;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	//default
	cudaEventRecord( start, 0 );
	transpose<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_out, dev_in, COLUMNS, ROWS );
	cudaEventRecord( stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed, start, stop );
	cudaMemcpy( &outmatrix[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
	//std::cout << "\nOUTPUT MATRIX - " << COLUMNS << " rows, " << ROWS << " columns" << std::endl;
	//printMatrix( &outmatrix[ 0 ], COLUMNS, ROWS );
	if( CSV ) {
		std::cout << "default," << ROWS << 'x' << COLUMNS << ',' << TILE_DIM << ',' 
				  << elapsed  << std::endl;  	
	}
	else {
		std::cout << "[default] elapsed time (ms): "  << elapsed << std::endl;
	}
	//coalesced
	cudaEventRecord( start, 0 );
	transposeCoalesced<<<BLOCKS, THREADS_PER_BLOCK>>>( dev_out, dev_in, COLUMNS, ROWS );
	cudaEventRecord( stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed, start, stop );
	cudaMemcpy( &outmatrix[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
	//std::cout << "\nOUTPUT MATRIX - " << COLUMNS << " rows, " << ROWS << " columns" << std::endl;
	//printMatrix( &outmatrix[ 0 ], COLUMNS, ROWS );
	if( CSV ) {
		std::cout << "coalesced," << ROWS << 'x' << COLUMNS << ',' << TILE_DIM << ',' 
				  << elapsed << std::endl;  	
	}
	else {
		std::cout << "[coalesced] elapsed time (ms): "  << elapsed  << std::endl;
	}

	cudaFree( dev_in );
	cudaFree( dev_out );
	cudaEventDestroy( start );
	cudaEventDestroy( stop  );

	return 0;
}
