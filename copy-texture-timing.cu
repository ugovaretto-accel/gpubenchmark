#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

typedef float real_t;

// TEXTURE: SINGLE PRECISION ONLY!
texture< real_t > texRef;


__global__ void copyFromTexture( real_t *odata, size_t offset ) {
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x ;
	odata[ i ] = tex1Dfetch( texRef, i + offset);
}

template < typename T >
T strTo( const char* str ) {
	if( !str ) throw std::runtime_error( "strToReal - NULL srting");
	std::istringstream is( str );
	T v = T();
	is >> v;
	return v;
}

const double GB = 1024 * 1024 * 1024;
double GBs( size_t numElements, double tms ) {
	return ( numElements * sizeof( real_t ) / GB ) / ( tms / 1000 );
}

//a.exe 4194304 0 16 1 16 128 csv
int main(int argc, char** argv ) {
		
	size_t NUM_ELEMENTS = 1024;
	size_t OFFSET_MIN = 0;
	size_t OFFSET_MAX = 0;
	size_t BLOCK_SIZE = 128;	
	bool CSV = false;

	if( argc < 2 || argc > 6 ) {
		std::cout << "usage: " << argv[ 0 ] << " <num elements> <offset> <stride> <block size> [csv]\n";
		std::cout << "  using default: num elements= " << NUM_ELEMENTS
				  << " offset= " << OFFSET_MIN << ',' << OFFSET_MAX 
				  << " block size= " << BLOCK_SIZE << std::endl;
	} else {
		NUM_ELEMENTS = strTo< size_t >( argv[ 1 ] );
		OFFSET_MIN = strTo< size_t >( argv[ 2 ] );
		OFFSET_MAX =  strTo< size_t >( argv[ 3 ] );
		BLOCK_SIZE = strTo< int >( argv[ 4 ] );
		if( argc == 6 ) { CSV = std::string( argv[ 5 ] ) == "csv"; }
	}

	const dim3 BLOCKS( ( NUM_ELEMENTS + BLOCK_SIZE - 1 ) / BLOCK_SIZE );
	const dim3 THREADS_PER_BLOCK( BLOCK_SIZE );
	const size_t TOTAL_ELEMENTS = NUM_ELEMENTS + OFFSET_MAX;
	const size_t SIZE = TOTAL_ELEMENTS * sizeof( real_t );

	std::vector< real_t > h_data( TOTAL_ELEMENTS, 1.f );


	// allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
    cudaArray* cu_array = 0;
    cudaMallocArray( &cu_array, &channelDesc, TOTAL_ELEMENTS, 1 ); 
    cudaMemcpyToArray( cu_array, 0, 0, &h_data[ 0 ], SIZE, cudaMemcpyHostToDevice);
    // set texture parameters
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTextureToArray( texRef, cu_array, channelDesc );

    real_t* dev_out = 0;
	cudaMalloc( &dev_out, SIZE );

	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop  = cudaEvent_t();
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
 
	
	for( size_t i = OFFSET_MIN; i <= OFFSET_MAX; ++i ) {
		float elapsed = 0.f;
		cudaEventRecord( start, 0 );
		copyFromTexture<<< BLOCKS, THREADS_PER_BLOCK >>>( dev_out, i );
		cudaEventRecord( stop, 0);
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed, start, stop );
		if( CSV ) {
			std::cout << NUM_ELEMENTS << ',' << i << ',' << BLOCK_SIZE << ',' 
					  << elapsed  << ',' << GBs( NUM_ELEMENTS, elapsed ) << std::endl;  	
		}
		else {
			std::cout << "elapsed time (ms): "  << elapsed << std::endl;
		}
	}
	
	cudaFree( dev_out );
	cudaEventDestroy( start );
	cudaEventDestroy( stop  );

	return 0;
}


/*
RESULTS:

------------------------------------------------
Penryn - Window 7 64bit - GTX 285 - CUDA 4.0 RC1
------------------------------------------------ 
C:\tmp>a.exe 65520 0 16 512 csv
65520,0,512,0.024448,9.98368
65520,1,512,0.01328,18.3796
65520,2,512,0.01312,18.6037
65520,3,512,0.013152,18.5585
65520,4,512,0.013408,18.2041
65520,5,512,0.012704,19.2129
65520,6,512,0.01296,18.8334
65520,7,512,0.013184,18.5134
65520,8,512,0.012864,18.974
65520,9,512,0.013408,18.2041
65520,10,512,0.015008,16.2634
65520,11,512,0.01328,18.3796
65520,12,512,0.013184,18.5134
65520,13,512,0.012896,18.9269
65520,14,512,0.013056,18.6949
65520,15,512,0.01472,16.5816
65520,16,512,0.012512,19.5078
*/