#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "Timer.h"

typedef float real_t;


__global__ void copy( real_t *odata, const real_t *idata, size_t offset, size_t stride ) {
	const size_t i = stride * ( 
		( blockIdx.y * blockDim.y + threadIdx.y ) * gridDim.x +
		( blockIdx.x * blockDim.x + threadIdx.x ) + offset );
	odata[ i ] = idata[ i ];
}


__global__ void initBuffer( real_t* in ) {
	const int i = threadIdx.x + blockDim.x * blockIdx.x; 
	in[ i ] = ( real_t ) i; 
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
	size_t STRIDE_MIN = 1;
	size_t STRIDE_MAX = 1;
	size_t BLOCK_SIZE = 128;	
	bool CSV = false;

	if( argc < 2 || argc > 8 ) {
		std::cout << "usage: " << argv[ 0 ] << " <num elements> <offset> <stride> <block size> [csv]\n";
		std::cout << "  using default: num elements= " << NUM_ELEMENTS
				  << " offset= " << OFFSET_MIN << ',' << OFFSET_MAX 
				  << " stride= " << STRIDE_MIN << ',' << STRIDE_MAX
				  << " block size= " << BLOCK_SIZE << std::endl;
	} else {
		NUM_ELEMENTS = strTo< size_t >( argv[ 1 ] );
		OFFSET_MIN = strTo< size_t >( argv[ 2 ] );
		OFFSET_MAX =  strTo< size_t >( argv[ 3 ] );
		STRIDE_MIN = strTo< size_t >( argv[ 4 ] );
		STRIDE_MAX = strTo< size_t >( argv[ 5 ] );
		BLOCK_SIZE = strTo< int >( argv[ 6 ] );
		if( argc == 8 ) { CSV = std::string( argv[ 7 ] ) == "csv"; }
	}

	const dim3 BLOCKS( ( NUM_ELEMENTS + BLOCK_SIZE - 1 ) / BLOCK_SIZE );
	const dim3 THREADS_PER_BLOCK( BLOCK_SIZE );
	const size_t TOTAL_ELEMENTS = std::max( NUM_ELEMENTS + OFFSET_MAX,  NUM_ELEMENTS * STRIDE_MAX );
	const size_t SIZE  = TOTAL_ELEMENTS * sizeof( real_t );

	real_t* dev_in = 0;
	real_t* dev_out = 0;

	cudaMalloc( &dev_in,  SIZE );
	cudaMalloc( &dev_out, SIZE );
	
	initBuffer<<< ( TOTAL_ELEMENTS + 256 - 1 ) / 256, 256 >>>( dev_in );
	
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop  = cudaEvent_t();
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
 
	
	for( size_t i = OFFSET_MIN; i <= OFFSET_MAX; ++i ) {
		float elapsed = 0.f;
		cudaEventRecord( start, 0 );
		copy<<< BLOCKS, THREADS_PER_BLOCK >>>( dev_out, dev_in, i, 1 );
		cudaEventRecord( stop, 0);
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed, start, stop );
		if( CSV ) {
			std::cout << NUM_ELEMENTS << ',' << i << ',' << 1 << ',' << BLOCK_SIZE << ',' 
					  << elapsed  << ',' << GBs( NUM_ELEMENTS, elapsed ) << std::endl;  	
		}
		else {
			std::cout << "offset copy - elapsed time (ms): "  << elapsed << " bandwidth: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;
		}
	}
	std::cout << '\n';
	for( size_t i = STRIDE_MIN; i <= STRIDE_MAX; ++i ) {
		float elapsed = 0.f;
		cudaEventRecord( start, 0 );
		copy<<< BLOCKS, THREADS_PER_BLOCK >>>( dev_out, dev_in, 0, i );
		cudaEventRecord( stop, 0);
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed, start, stop );
		if( CSV ) {
			std::cout << NUM_ELEMENTS << ',' << 0 << ',' << i << ',' << BLOCK_SIZE << ','
					  << elapsed  << ',' << GBs( NUM_ELEMENTS, elapsed ) << std::endl;  	
		}
		else {
			std::cout << "stride copy - elapsed time (ms): "  << elapsed <<  " bandwidth: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;
		}
	}

	std::cout << std::endl;

	if( !CSV )	
	{
		cudaThreadSynchronize();
		std::vector< real_t > h_out( TOTAL_ELEMENTS );
		std::vector< real_t > h_out_cpy( TOTAL_ELEMENTS );
		float elapsed = 0.f;
		cudaEventRecord( start, 0);
		cudaMemcpy( &h_out[ 0 ], dev_out, SIZE, cudaMemcpyDeviceToHost );
		cudaEventRecord( stop, 0);
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsed, start, stop );
		std::cout << "cudaMemcopy (ms): " << elapsed << " bandwidth: " << GBs( TOTAL_ELEMENTS, elapsed ) << std::endl;
		
		Timer t;
		t.Start();
		memmove( &h_out_cpy[ 0 ], &h_out[ 0 ], SIZE );
		double e = t.Stop();
		std::cout << "memmove (ms): " << e << " bandwidth: " << GBs( TOTAL_ELEMENTS, e ) << std::endl;
		
		t.Start();
		memcpy( &h_out_cpy[ 0 ], &h_out[ 0 ], SIZE );
		std::cout << "memcpy (ms): " << e << " bandwidth: " << GBs( TOTAL_ELEMENTS, e ) << std::endl;

		t.Start();
		for( int i = 0; i != TOTAL_ELEMENTS; ++i ) h_out_cpy[ i ] = h_out[ i ];
		e = t.Stop();
		std::cout << "for (ms): " << e << " bandwidth: " << GBs( TOTAL_ELEMENTS, e ) << std::endl;

		t.Start();
		std::copy( h_out.begin(), h_out.end(), h_out_cpy.begin() );
		e = t.Stop();
		std::cout << "std::copy (ms): " << e << " bandwidth: " << GBs( TOTAL_ELEMENTS, e ) << std::endl;

		t.Start();
		for( real_t  *p1 = &h_out[ 0 ], *p2 = &h_out_cpy[ 0 ]; p1 != &h_out[ 0 ] + TOTAL_ELEMENTS; ++p1, ++p2 ) *p2 = *p1; 
		e = t.Stop();
		std::cout << "trivial (ms): " << e  << " bandwidth: " << GBs( TOTAL_ELEMENTS, e ) << std::endl;
		
		//for multidimensional arrays use
		//for( pi1 = buffer1, pi2 = buffer2; pi1 < buffer1 + bfsize1; pi1 += rowstride, pi2 += rowstride ) {
		//	for( pj1 = pi1, pj2 = pi2; pj1 < pi1 + columnsize; ++pj1, ++pj2 ) {
		//		*pj2 = *pj1;
		//	}
		//}

	}
	
	cudaFree( dev_in );
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
C:\tmp>a.exe 4194304 0 16 1 16 128 csv
offset
4194304,0,1,128,0.262976,59.4161
4194304,1,1,128,0.43792,35.68
4194304,2,1,128,0.43872,35.615
4194304,3,1,128,0.43664,35.7846
4194304,4,1,128,0.438688,35.6176
4194304,5,1,128,0.437568,35.7087
4194304,6,1,128,0.439456,35.5553
4194304,7,1,128,0.436864,35.7663
4194304,8,1,128,0.389952,40.069
4194304,9,1,128,0.437376,35.7244
4194304,10,1,128,0.436544,35.7925
4194304,11,1,128,0.43776,35.6931
4194304,12,1,128,0.438464,35.6358
4194304,13,1,128,0.437568,35.7087
4194304,14,1,128,0.437216,35.7375
4194304,15,1,128,0.4376,35.7061
4194304,16,1,128,0.261088,59.8457
stride
4194304,0,1,128,0.259392,60.237
4194304,0,2,128,0.499072,31.3081
4194304,0,3,128,0.775168,20.1569
4194304,0,4,128,1.04979,14.8839
4194304,0,5,128,1.31629,11.8705
4194304,0,6,128,1.58934,9.8311
4194304,0,7,128,1.86986,8.35626
4194304,0,8,128,2.16294,7.22395
4194304,0,9,128,2.34682,6.65796
4194304,0,10,128,2.59024,6.03226
4194304,0,11,128,2.7656,5.64977
4194304,0,12,128,3.08483,5.06511
4194304,0,13,128,3.24205,4.81948
4194304,0,14,128,3.54125,4.41229
4194304,0,15,128,3.81421,4.09653
4194304,0,16,128,4.20851,3.71271



*/
