#include <stdio.h>
#include <malloc.h>
#include <sys/timeb.h>
#include <time.h>

// The following includes for MS VC++ intelliSense to flag problems
// in syntax of CUDA that NVCC automatically includes. E.g., "__global__".
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

// NO COALESCING -- ELEMENTS IN EACH HALF WARP OUT OF SEQUENCE.
// NOTE, ON FERMI CARDS, THIS DOESN'T MATTER BECAUSE IT USES CACHING.
__device__ int reverse[32] = { 0,
 2,  1,  3,  4,  5,  6,  7,  8,  9, 10,
11, 12, 13, 14, 15, 16, 18, 17, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

__global__ void no_coalesce(int * data, int n, int iter)
{
    // assume one block of size 32.
    int idx = threadIdx.x;
    __shared__ int sr[32];
    sr[idx] = reverse[idx];
    __syncthreads();
    for (int i = 0; i < iter; ++i)
        data[sr[idx]] += n;
}

// NO COALESCING -- LAST ELEMENT FETCHED IS NOT CONTIGUOUS.
__device__ int extended[32] = { 0,
 1,  333,  3,  4,  5,  6,  7,  8,  9, 10,
11, 12, 13, 14, 15, 566, 17, 18, 19, 20,
21, 22, 222, 24, 25, 26, 27, 28, 29, 30, 444};

__global__ void no_coalesce2(int * data, int n, int iter)
{
    // assume one block of size 32.
    int idx = threadIdx.x;
    __shared__ int ex[32];
    ex[idx] = extended[idx];
    __syncthreads();
    for (int i = 0; i < iter; ++i)
        data[ex[idx]] += n;
}

__device__ int forward[32] = { 0,
 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

__global__ void coalesce(int * data, int n, int iter)
{
    // assume one block of size 32.
    int idx = threadIdx.x;
    __shared__ int sf[32];
    sf[idx] = forward[idx];
    __syncthreads();
    for (int i = 0; i < iter; ++i)
        data[sf[idx]] += n;
}

int main(int argc, char**argv)
{
    argc--; argv++;

    // First argv is an int, cuda device number.
    int rvdev = cudaSetDevice(atoi(*argv));

    // Setup for "in" host array.
    int n = 32;  // number of elements of arrays that are changed by kernel.
    int extended = 5000; // actual length of array.

    int * in = (int*)malloc(extended * sizeof(int));
    for (int i = 0; i < extended; ++i)
        in[i] = 0;

    // Setup for "out" host array.
    int * out = (int*)malloc(extended * sizeof(int));

    // Timers.
    struct _timeb  t1;
    struct _timeb  t2;
    struct _timeb  t3;
    struct _timeb  t4;

    /////////////////////////////////////////////
    // Test no_coalescing example.
    /////////////////////////////////////////////
    printf("Starting GPU test v1 ...\n");
    _ftime(&t1);
    int * din;
    int rv1 = cudaMalloc(&din, extended * sizeof(int));
    _ftime(&t2);
    int rv2 = cudaMemcpy(din, in, extended * sizeof(int), cudaMemcpyHostToDevice);
    _ftime_s(&t3);
    int kernel_calls = 1;
    int internal_iters = 10000000;
    int block_size = 32;
    int blocks = 1;
    dim3 block(block_size);
    dim3 grid(blocks);
    no_coalesce<<<grid, block>>>(din, n, internal_iters);
    cudaThreadSynchronize();
    int rv3 = cudaGetLastError();
    if (rv3)
        printf("last error %d\n", rv1);
    _ftime(&t4);
    printf("N Time t4-t3 %f\n", (double)(t4.time - t3.time + ((double)(t4.millitm - t3.millitm))/1000));
    int rv4 = cudaMemcpy(out, din, extended * sizeof(int), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < block_size; ++i)
    //  printf("%d %d\n", i, out[i]);

    /////////////////////////////////////////////
    // Test coalescing example.
    /////////////////////////////////////////////
    // Reset device "in" array.
    int rv5 = cudaMemcpy(din, in, extended * sizeof(int), cudaMemcpyHostToDevice);
    _ftime_s(&t3);
    coalesce<<<grid, block>>>(din, n, internal_iters);
    cudaThreadSynchronize();
    int rv6 = cudaGetLastError();
    if (rv6)
        printf("last error %d\n", rv1);
    _ftime(&t4);
    printf("C Time t4-t3 %f\n", (double)(t4.time - t3.time + ((double)(t4.millitm - t3.millitm))/1000));
    int rv7 = cudaMemcpy(out, din, extended * sizeof(int), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < block_size; ++i)
    //  printf("%d %d\n", i, out[i]);


    /////////////////////////////////////////////
    // Test no_coalescing2 example.
    /////////////////////////////////////////////
    // Reset device "in" array.
    int rv8 = cudaMemcpy(din, in, extended * sizeof(int), cudaMemcpyHostToDevice);
    _ftime_s(&t3);
    no_coalesce2<<<grid, block>>>(din, n, internal_iters);
    cudaThreadSynchronize();
    int rv9 = cudaGetLastError();
    if (rv9)
        printf("last error %d\n", rv1);
    _ftime(&t4);
    printf("E Time t4-t3 %f\n", (double)(t4.time - t3.time + ((double)(t4.millitm - t3.millitm))/1000));
    int rv10 = cudaMemcpy(out, din, extended * sizeof(int), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < block_size; ++i)
    //  printf("%d %d\n", i, out[i]);

    cudaFree(din);

    return 0;
}

/******
This code accesses an array in three different patterns. For an explanation of the rules for coalescing (of global memory), see section G.3.2 of the NVIDIA CUDA C Programming Guide (PG).

"coalesce" accesses a 128-byte array sequentially. First, all data is accessed in words (from the PG, "The size of the words accessed by the threads must be 4, 8, or 16 bytes;", and in the code "data[sf[idx]] += n;", "data" is an array of 4-byte values). Next, all data for each half warp is in 64 bytes ("If this size is: 4, all 16 words must lie in the same 64-byte segment", and in the code "data[sf[idx]] += n", "data" is only accessed from addresses &data[0] through &data[15] in the first half warp, &data[16] through &data[31] in the first half warp). Finally, all data is accessed sequentially from &data[0] through &data[15] in the first half warp, &data[16] through &data[31] in the first half warp. The array "forward" specifies the indices of the access. Notice that you must run this on a CUDA 1.0 or 1.1 device like GeForce 9800 GT.

"no_coalesce" accesses a 128-byte array in a non-sequential manner, so coalescing does not occur. While all data is accessed in words, all data for each half warp is in 64 bytes, "data" is NOT accessed sequentially for each half warp. In the first half warp, "data" is accessed in the order &data[0], &data[2], &data[1], &data[3], &data[4], ..., &data[15]; in the second half warp, "data" is accessed in the order &data[16], &data[18], &data[17], &data[19], ..., &data[31]. This violates the rule that say it must be in order.

"no_coalesce2" accesses the array in a non-contiguous manner, so coalescing does not occur. While all data is accessed in words, all data for each half warp is in 64 bytes, the data is NOT accessed in a contiguous array of memory for each half warp. The first two accesses are contiguous (&data[0], &data[1]), the third is not (&data[333]).

On my GeForce 9800 GT, the output from this example is:

Starting GPU test v1 ...
N Time t4-t3 1.156000
C Time t4-t3 0.800000
E Time t4-t3 1.157000

These times display coalescing for the 2nd case ("C", i.e., "coalesce") only. The other two do not have coalesce access of global memory.

However, if I run this on my GeForce GTX 470, the output is:

Starting GPU test v1 ...
N Time t4-t3 0.626000
C Time t4-t3 0.626000
E Time t4-t3 0.626000

These times display no difference with regards to the access pattern. Coalesced global memory access does not exist on 2.0 devices. (Or, if it does--which I don't believe it does--it is a much more complicated access pattern, and so NVIDIA does not want to describe it in the Programming Guide.)



*******/