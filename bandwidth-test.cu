#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <malloc.h>
#include "Timer.h"

typedef float real_t;



void* aligned_malloc( size_t size, size_t alignment ) {
    assert( sizeof( size_t ) == sizeof( void* ) );
    const size_t OFFSET = sizeof( size_t ) > alignment ? sizeof( size_t ) : alignment;
    char* ptr = ( char* ) malloc( size + OFFSET );
    char* buffer_start = ptr + OFFSET;
    if( ( size_t ) buffer_start  % alignment != 0 ) {
        buffer_start += alignment - ( ( size_t ) buffer_start ) % alignment;
    }
    size_t* start_pointer = ( size_t* ) ( buffer_start - sizeof( size_t ) );
    *start_pointer = ( size_t ) ptr;  
    std::cout << *start_pointer << std::endl;        
    if( ( size_t ) buffer_start % alignment != 0 ) std::cerr << "ERROR" << std::endl;
    return buffer_start;
}

void aligned_free( void* ptr ) {
    std::cout <<  *( ( size_t* ) ( (char*) ptr - sizeof( size_t ) ) ) << std::endl;
    free( ( void* ) *( (size_t* ) ( (char*) ptr - sizeof( size_t ) ) ) ) ; 
}


size_t findPageLockedLimit( size_t prev = 65536 ) {
    float* p = 0;
    const cudaError_t e = cudaHostAlloc( &p, prev, cudaHostAllocDefault );
    if( e != cudaSuccess ) return prev / 2;
    cudaFreeHost( p );
    return findPageLockedLimit( 2 * prev );
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
double GBs( size_t numElements, double tms ) {
    return ( ( numElements * sizeof( real_t ) ) / GB ) / ( tms / 1000 );
}

//a.exe 4194304 0 16 1 16 128 csv
int main(int argc, char** argv ) {

    size_t NUM_ELEMENTS = 64 * 1024  *1024;
    bool CSV = false;
        
    if( argc < 2 || argc > 3 ) {
        std::cout << "usage: " << argv[ 0 ] << " <num elements>  [csv]\n";
        std::cout << "  using default: num elements= " << NUM_ELEMENTS << std::endl;
    } else {
        NUM_ELEMENTS = strTo< size_t >( argv[ 1 ] );
        std::cout << "NUM ELEMENTS: " << NUM_ELEMENTS << ' ' <<
                      double( NUM_ELEMENTS * sizeof( real_t ) ) / ( 1024*1024 ) <<
                      " MB" << std::endl;
        if( argc == 3 ) { CSV = std::string( argv[ 2 ] ) == "csv"; }
    }

    if( cudaSetDevice( 0 ) != cudaSuccess ) {
        std::cerr << "Cannot set device" << std::endl;
        return 1;
    }
cudaDeviceReset();
cudaDeviceSynchronize();
    cudaSetDeviceFlags( cudaDeviceMapHost );
#if 0       
    std::cout <<     "\nMax power of two page lockable size:                      " 
              << findPageLockedLimit() / ( 1024 * 1024 ) << "MB" << std::endl;
#endif  
            
    const size_t SIZE = sizeof( real_t ) * NUM_ELEMENTS;            
            
    cudaEvent_t start = cudaEvent_t();
    cudaEvent_t stop  = cudaEvent_t();
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    float elapsed = 0.f;
    
    //
    {
        real_t* dev_in, *dev_out;
        cudaMalloc( &dev_in, SIZE );
        cudaMalloc( &dev_out, SIZE );
        
        cudaEventRecord( start );
            cudaMemcpy( dev_out, dev_in, SIZE, cudaMemcpyDeviceToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to device - cudaMemcpyDeviceToDevice:              "  << 2 * GBs( NUM_ELEMENTS, elapsed ) << std::endl;
        
        cudaFree( dev_in );
        cudaFree( dev_out ); 
    }
    //
    {
        real_t *dev;
        //std::vector< real_t > host( NUM_ELEMENTS );
        real_t* host = (real_t*) memalign( 128, SIZE );
        if( (size_t ) host % 16 == 0 ) std::cout << "ALIGNED" << std::endl;
        cudaMalloc( &dev, SIZE );
        
        cudaEventRecord( start );
        cudaMemcpy( dev, &host[ 0 ], SIZE, cudaMemcpyHostToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaMalloc:                              " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;
        
        cudaEventRecord( start );
            cudaMemcpy( &host[ 0 ], dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );  
        std::cout << "\nDevice to host - cudaMalloc:                              " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;

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
        cudaEventRecord( start );
            cudaMemcpy( dev, host, SIZE, cudaMemcpyHostToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaAllocHost:                           " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;
        
        cudaEventRecord( start );
            cudaMemcpy( host, dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to host - cudaAllocHost:                           " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;

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
        
        cudaEventRecord( start );
            cudaMemcpy( dev, host, SIZE, cudaMemcpyHostToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaAllocHost - portable:                " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;   

        cudaEventRecord( start );   
            cudaMemcpy( host, dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to host - cudaAllocHost - portable:                " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;

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
        
        cudaEventRecord( start );
            cudaMemcpy( dev, host, SIZE, cudaMemcpyHostToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaAllocHost - mapped:                  " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;   
        
        cudaEventRecord( start );
            cudaMemcpy( host, dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to host - cudaAllocHost - mapped:                  " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;       
        
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
        
        cudaEventRecord( start );
            cudaMemcpy( dev, mapped_host, SIZE, cudaMemcpyDeviceToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaAllocHost/device to device - mapped: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;   
            
        cudaEventRecord( start );       
            cudaMemcpy( host, dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to host - cudaAllocHost/device to device - mapped: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;       
        
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

        cudaEventRecord( start );
            cudaMemcpy( dev, host, SIZE, cudaMemcpyHostToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaAllocHost - write combining:         " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;   
        
        cudaEventRecord( start );
            cudaMemcpy( host, dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to host - cudaAllocHost - write combining:         " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;       
        
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
        
        cudaEventRecord( start );
            cudaMemcpy( dev, mapped_host, SIZE, cudaMemcpyDeviceToDevice );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nHost to device - cudaAllocHost - mapped, write combining: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;   

        cudaEventRecord( start );
            cudaMemcpy( host, dev, SIZE, cudaMemcpyDeviceToHost );
        cudaEventRecord( stop );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &elapsed, start, stop );
        std::cout << "\nDevice to host - cudaAllocHost - mapped, write combining: " << GBs( NUM_ELEMENTS, elapsed ) << std::endl;   


        cudaFree( dev );
        cudaFreeHost( host );
    }

    #if CUDART_VERSION >= 4000
    //
    {
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
    //
    {
        void *hostSrc, *hostDest;
        hostSrc  = malloc( NUM_ELEMENTS * sizeof( real_t ) );
                hostDest = malloc( NUM_ELEMENTS * sizeof( real_t ) );
                timespec t1,t2;
                //timeval t1, t2;
                double elapsedTime;
                //gettimeofday( &t1, 0 );
                clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &t1 );
                memcpy( hostDest, hostSrc, NUM_ELEMENTS * sizeof( real_t ) );
                clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &t2 );
                //gettimeofday( &t2, 0 );
                //elapsedTime = ( t2.tv_sec - t1.tv_sec ) * 1000.0 +
                //              ( t2.tv_usec - t1.tv_usec ) / 1000.0;     
                elapsedTime = ( t2.tv_sec - t1.tv_sec ) * 1000.0 +
                              ( t2.tv_nsec - t1.tv_nsec ) / 1000000;

                std::cout << "\nHost to Host: " << 2 * GBs( NUM_ELEMENTS, float( elapsedTime ) ) << std::endl;  

        free( hostSrc  );
        free( hostDest );
    }
    {
        void *hostSrc, *hostDest;
        hostSrc  = aligned_malloc( NUM_ELEMENTS * sizeof( real_t ), 256 );
                hostDest = aligned_malloc( NUM_ELEMENTS * sizeof( real_t ), 256 );
                timespec t1,t2;
                //timeval t1, t2;
                double elapsedTime;
                //gettimeofday( &t1, 0 );
                clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &t1 );
                memcpy( hostDest, hostSrc, NUM_ELEMENTS * sizeof( real_t ) );
                clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &t2 );
                //gettimeofday( &t2, 0 );
                //elapsedTime = ( t2.tv_sec - t1.tv_sec ) * 1000.0 +
                //              ( t2.tv_usec - t1.tv_usec ) / 1000.0;     
                elapsedTime = ( t2.tv_sec - t1.tv_sec ) * 1000.0 +
                              ( t2.tv_nsec - t1.tv_nsec ) / 1000000;

                std::cout << "\nHost to Host - aligned: " << 2 * GBs( NUM_ELEMENTS, float( elapsedTime ) ) << std::endl;  

        aligned_free( hostSrc  );
        aligned_free( hostDest );
    }
    return 0;
}
