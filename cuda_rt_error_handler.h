#pragma once
#ifndef CUDART_ERROR_HANDLER_
#define CUDART_ERROR_HANDLER_
///\file cudart_error_handler.h Error handler for CUDA run-time API calls

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

inline void HandleCUDAError( cudaError_t err,
                             const char *file,
                             int line,
                             const char* msg = 0 )
{
    if( err != cudaSuccess )
	{
		std::ostringstream ss;
		ss << ( msg != 0 ? msg : "" ) << " File: " << file << ", Line: " << line << ", Error: " << ::cudaGetErrorString( err );
		throw std::runtime_error( ss.str() );
    }
}
#define HANDLE_CUDA_ERROR( err ) ( HandleCUDAError( err, __FILE__, __LINE__ ) )

#define LAUNCH_CUDA_KERNEL( k ) \
  k; \
  HandleCUDAError( ::cudaGetLastError(), __FILE__, __LINE__, "(Kernel launch)" ); \

#endif //CUDART_ERROR_HANDLER_
