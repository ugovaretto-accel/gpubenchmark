#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cassert>

typedef float real_t;


size_t searchPageLockedLimit( size_t m = 0xffff, size_t M = 0x3fffffff ) {
	assert( m <= M );
	const size_t h = ( M + m ) / 2;
	if( h == m || h > size_t(1E9) ) return h;
	void* p = 0;
	if( cudaHostAlloc( &p, h, cudaHostAllocDefault ) == cudaSuccess ) {
		cudaFreeHost( p );
		return searchPageLockedLimit( h, M );
	}
	return searchPageLockedLimit( m, h );
}


size_t findPow2PageLockedLimit( size_t s = 65536 ) {
	void* p = 0;
	cudaError_t e = cudaSuccess;
	while( e == cudaSuccess && s > 0 ) {
		e = cudaHostAlloc( &p, s, cudaHostAllocDefault );
		if( e == cudaSuccess ) {
			cudaFreeHost( p );
			return s;
		}
		s >>= 1;
	}
	return s;
}


template < typename T >
T strTo( const char* str ) {
	if( !str ) throw std::runtime_error( "strTo<T> - NULL srting");
	std::istringstream is( str );
	T v = T();
	is >> v;
	return v;
}

const double GB = 1024 * 1024 * 1024;
double GBs( size_t bytes, double tms ) {
	return ( bytes / GB ) / ( tms / 1000 );
}


cudaEvent_t start = cudaEvent_t();
cudaEvent_t stop  = cudaEvent_t();


cudaError_t memCpy( void* target, const void* src, size_t bytes, cudaMemcpyKind flags, const char* msg ) {
	float elapsed = 0.f;
	cudaError_t e = cudaEventRecord( start ); if( e != cudaSuccess ) return e;
 	e = cudaMemcpy( target, src, bytes, flags ); if( e != cudaSuccess ) return e;
 	e = cudaEventRecord( stop ); if( e != cudaSuccess ) return e;
 	e = cudaEventSynchronize( stop ); if( e != cudaSuccess ) return e;
 	e = cudaEventElapsedTime( &elapsed, start, stop ); if( e != cudaSuccess ) return e;
 	std::cout << bytes << ',' << msg << ','  << GBs( bytes, elapsed ) << ',' << elapsed << std::endl;
 	return e;
 }


size_t nextStep( size_t prev, size_t val, bool mul ) {
	return mul ? val * prev : prev + val;
}


int main(int argc, char** argv ) {

	size_t beginSize = 0;
	size_t endSize   = 0;
	size_t step      = 0;
	bool   mul       = false;
	int    dev_id    = 0;
			
	if( argc < 5 ) {
		std::cout << "usage: " << argv[ 0 ] << " <min num elements>  <max num elements> <step> [device id]\n";
		return 0;
	} else {		
		beginSize = strTo< size_t >( argv[ 1 ] );
		endSize   = strTo< size_t >( argv[ 2 ] );
		step 	  = strTo< size_t >( argv[ 3 ] );
		if( argc > 4 ) dev_id = strTo< int >( argv[ 4 ] ); 
	}

	if( cudaSetDevice( dev_id ) != cudaSuccess ) {
		std::cerr << "Cannot set device " << dev_id << std::endl;
		return 1;
	}

	cudaSetDeviceFlags( cudaDeviceMapHost );
	
        #if 0
	//size_t pll = findPow2PageLockedLimit( 1024 * 1024 * 1024 );
	pll = searchPageLockedLimit( pll, 2 * pll );	
	std::cout <<  "Max power of two page-lockable size (<= 1GB): " 
	          <<  double( pll ) / ( 1024 * 1024 ) << "MB" << std::endl;
	#endif	
	
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
 	
 	for( size_t s = beginSize; s < endSize; s = nextStep( s, step, mul ) ) {
 		const size_t NUM_ELEMENTS = s;
 		const size_t SIZE = NUM_ELEMENTS * sizeof( real_t );
 		{
 			real_t* dev_in, *dev_out;
 			cudaMalloc( &dev_in, SIZE );
 			cudaMalloc( &dev_out, SIZE );
 			memCpy( dev_out, dev_in, SIZE, cudaMemcpyDeviceToDevice, "Device to Device"); 		
 			cudaFree( dev_in );
 			cudaFree( dev_out ); 
 		}
		//
		{
			real_t *dev;
			std::vector< real_t > host( NUM_ELEMENTS );
			cudaMalloc( &dev, SIZE );
			memCpy( dev, &host[ 0 ], SIZE, cudaMemcpyHostToDevice, "Host to Device"  );
			memCpy( &host[ 0 ], dev, SIZE, cudaMemcpyDeviceToHost, "Device to host"  );
			cudaFree( dev );
		}
		//
		{
			real_t *dev, *host;
			cudaMalloc( &dev, SIZE );
			const cudaError_t e =  cudaHostAlloc( &host, SIZE, cudaHostAllocDefault /*same as cudaMallocHost*/ );
			if( e != cudaSuccess ) {
				std::cerr << "Error - cudaHostAlloc" << std::endl;
				return 1;
			}
			memCpy( dev, host, SIZE, cudaMemcpyHostToDevice, "Host to Device - cudaHostAllocDefault" );
			memCpy( host, dev, SIZE, cudaMemcpyDeviceToHost, "Device to Host - cudaHostAllocDefault" );
			cudaFree( dev );
			cudaFreeHost( host );
		}
		//
		{
			real_t *dev, *host;
			cudaMalloc( &dev, SIZE );
			const cudaError_t e = cudaHostAlloc( &host, SIZE, cudaHostAllocPortable );
			if( e != cudaSuccess ) {
				std::cerr << "Error - cudaHostAlloc" << std::endl;
				return 1;
			}
			memCpy( dev, host, SIZE, cudaMemcpyHostToDevice,  "Host to Device - cudaHostAllocPortable" );
			memCpy( host, dev, SIZE, cudaMemcpyDeviceToHost,   "Device to Host - cudaHostAllocPortable" );
			cudaFree( dev );
			cudaFreeHost( host );
		}
		//
		{
			real_t *dev, *host;
			cudaMalloc( &dev, SIZE );
			const cudaError_t e = cudaHostAlloc( &host, SIZE, cudaHostAllocMapped );
			if( e != cudaSuccess ) {
				std::cerr << "Error - cudaHostAlloc - mapped" << std::endl;
				return 1;
			}
			memCpy( dev, host, SIZE, cudaMemcpyHostToDevice, "Host to Device - cudaHostAllocMapped" );
			memCpy( host, dev, SIZE, cudaMemcpyDeviceToHost, "Device to Host - cudaHostAllocMapped" );
			cudaFree( dev );
			cudaFreeHost( host );
		}
		//
		{
			real_t *dev, *host, *mapped_host;
			cudaMalloc( &dev, SIZE );
			const cudaError_t e  = cudaHostAlloc( &host, SIZE, cudaHostAllocMapped );
			if( e != cudaSuccess ) {
				std::cerr << "Error - cudaHostAlloc - mapped" << std::endl;
				return 1;
			}
			cudaHostGetDevicePointer( &mapped_host, host, 0 /*has to be zero "for now" */ );
			memCpy( dev, mapped_host, SIZE, cudaMemcpyDeviceToDevice, "Device(on host) to Device - cudaHostAllocMapped" );
			memCpy( mapped_host, dev, SIZE, cudaMemcpyDeviceToDevice, "Device to Device(on host) - cudaHostAllocMapped" );
			cudaFree( dev );
			cudaFreeHost( host );
		}
		//
		{
			real_t *dev, *host;
			cudaMalloc( &dev, SIZE );
			const cudaError_t e = cudaHostAlloc( &host, SIZE, cudaHostAllocWriteCombined );
			if( e != cudaSuccess ) {
				std::cerr << "Error - cudaHostAlloc - write combining" << std::endl;
				return 1;
			}
			memCpy( dev, host, SIZE, cudaMemcpyHostToDevice, "Host to Device - cudaHostAllocWriteCombined" );
			memCpy( host, dev, SIZE, cudaMemcpyDeviceToHost, "Device to Host - cudaHostAllocWriteCombined" );
			cudaFree( dev );
			cudaFreeHost( host );
		}
		//
		{
			real_t *dev, *host, *mapped_host;
			cudaMalloc( &dev, SIZE );
			const cudaError_t e = cudaHostAlloc( &host, SIZE, cudaHostAllocMapped | cudaHostAllocWriteCombined );
			if( e != cudaSuccess ) {
				std::cerr << "Error - cudaHostAlloc - mapped | write combining" << std::endl;
				return 1;
			}
			cudaHostGetDevicePointer( &mapped_host, host, 0 /*has to be zero "for now" */ );
			memCpy( dev, mapped_host, SIZE, cudaMemcpyDeviceToDevice, "Device(on host) to Device - cudaHostAllocMapped | cudaHostAllocWriteCombined" );
			memCpy( mapped_host, dev, SIZE, cudaMemcpyDeviceToDevice, "Device to Device(on host) - cudaHostAllocMapped | cudaHostAllocWriteCombined" );
			cudaFree( dev );
			cudaFreeHost( host );
		}

		#if CUDART_VERSION >= 4000
		//
		{
                        float elapsed = float();
		 	real_t *dev, *mapped_host;
		 	cudaMalloc( &dev, SIZE );
		 	std::vector< real_t > host( NUM_ELEMENTS );	

		 	const cudaError_t e = cudaHostRegister( &host[ 0 ], SIZE, cudaHostAllocMapped );
		 	if( e != cudaSuccess ) {
		 		std::cerr << "Error - cudaHostRegister - mapped" << std::endl;
		 		return 1;
		 	}
		 	cudaHostGetDevicePointer( &mapped_host, &host[ 0 ], 0 /*has to be zero "for now" */ );
			
		 	cudaEventRecord( start );
		 		cudaMemcpy( dev, mapped_host, SIZE, cudaMemcpyDeviceToDevice );
		 	cudaEventRecord( stop );
	 	 	cudaEventSynchronize( stop );
	 	 		cudaEventElapsedTime( &elapsed, start, stop );
		 	std::cout << "\nHost to device - cudaHostRegister - mapped: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;	

		 	cudaEventRecord( start );
		 		cudaMemcpy( &host[ 0 ], dev, SIZE, cudaMemcpyDeviceToHost );
		 	cudaEventRecord( stop );
	 	 		cudaEventSynchronize( stop );
	 	 		cudaEventElapsedTime( &elapsed, start, stop );
		 	std::cout << "\nDevice to host - cudaHostRegister - mapped: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;	

		 	cudaFree( dev );
		 	cudaHostUnregister( &host[ 0 ] );
		 }
		 #endif
	}

	return 0;
}
