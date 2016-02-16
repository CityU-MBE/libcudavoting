/*
 * =====================================================================================
 *
 *       Filename:  CudaVoting.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/09/2012 04:45:11 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */
#include <iostream>
//#include <cutil_math.h>
//#include <cutil_inline.h>
//#include <cutil_inline_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <assert.h>
#include <string.h>
#include <cuda.h>

#include "CudaVoting.h"

//#define check_memory_size
#define FDIV(X,Y) __fdividef((X),(Y)) //TODO: use it

//#define INT_ENDING 1000000 // cuda has 32-bit int definition: 4294,k,k
#define INT_ENDING 10000
#define use_fast_float_math

//#define BLOCK_DIM 512
#define BLOCK_DIM 256
#define BLOCK_DIM_G 128 // for pointgrid generation


#define cutilSafeCall 


using namespace std;

#include "CudaTools.cu" // must be included first
#include "CudaSparseVoting_kernel.cu"
#include "CudaDenseVoting_kernel.cu"
#include "CudaMat33_kernel.cu"
#include "CudaTensorSplit.cu"

namespace cudavoting{

    ///////////////////////////////////////////////////////////
    // Hardware check and selection
    //
    // redefined for CUDA 4.2
    //#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
    inline void __checkCudaErrors(cudaError err, const char *file, const int line )
    {
        if(cudaSuccess != err)
        {
            fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);        
        }
    }

    // This will output the proper error string when calling cudaGetLastError
    #define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
    inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }
    // General GPU Device CUDA Initialization
    int gpuDeviceInit(int devID)
    {
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));

        if (deviceCount == 0)
        {
            fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }

        if (devID < 0)
          devID = 0;

        if (devID > deviceCount-1)
        {
            fprintf(stderr, "\n");
            fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
            fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
            fprintf(stderr, "\n");
            return -devID;
        }

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

        if (deviceProp.major < 1)
        {
            fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
            exit(-1);                                                  
        }

        checkCudaErrors( cudaSetDevice(devID) );
        printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

        return devID;
    }
    // This function returns the best GPU (with maximum GFLOPS)
    int gpuGetMaxGflopsDeviceId()
    {
        int current_device     = 0, sm_per_multiproc  = 0;
        int max_compute_perf   = 0, max_perf_device   = 0;
        int device_count       = 0, best_SM_arch      = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceCount( &device_count );

        // Find the best major SM Architecture GPU device
        while (current_device < device_count)
        {
            cudaGetDeviceProperties( &deviceProp, current_device );
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            }
            current_device++;
        }

        // Find the best CUDA capable GPU device
        current_device = 0;
        while( current_device < device_count )
        {
            cudaGetDeviceProperties( &deviceProp, current_device );
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if( compute_perf  > max_compute_perf )
            {
                // If we find GPU with SM major > 2, search only these
                if ( best_SM_arch > 2 )
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
            ++current_device;
        }
        return max_perf_device;
    }

    // Initialization code to find the best CUDA Device
    int findCudaDevice()
    {
        cudaDeviceProp deviceProp;
        int devID = 0;
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
        return devID;
    }

    // functions
    int routineCheck()
    {
        int devID = findCudaDevice();
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        return devID;
    }

    unsigned int freeDeviceMemorySize() {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        printf("free = %d , total = %d\n", (unsigned int)free, (unsigned int)total);
        return free;
    }
    // @h_fieldarray : output init status of outputing field
    // @h_grids: input center positions of all grids
    // @h_points : input coords of points
    // @h_sparsestick: input sparse stick, eigen values and eigen vectors
    // @sigma : input sigma size
    // @numPoints : input number of points
    // @h_logg : output log
    extern "C"
    void CudaVoting::denseGridStickVoting(float3 * h_fieldarray, float3 * h_grids,
                    const float3 * h_points, const float4 * h_sparsestick,
                    const float sigma, const unsigned int numGrids, const int numPoints, int2 * h_log)
    {
    cudaEvent_t start_event, start_event0, stop_event, stop_event0;
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );
    cutilSafeCall( cudaEventCreate(&start_event0) );
    cutilSafeCall( cudaEventCreate(&stop_event0) );
    float elapsed_time0, elapsed_time;
    cudaEventRecord(start_event0, 0);

        // select device and initialize CUDA environment
        cuInit(0);
//        routineCheck();
#ifdef check_memory_size
        // check whether the field size to be created fit in mem
        size_t memsize = sizeof(float)*4*4*numPoints; //(3x3 + 3x1 = 3x4) + 4x1
        CUdevice cudaDevice;  
        CUresult result = cuDeviceGet(&cudaDevice, 0);  
        if(result!= CUDA_SUCCESS)  
        {  
            cout << "Error fetching cuda device";  
            return ;  
        }  
        CUcontext cudaContext;  
        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
        if(result != CUDA_SUCCESS)  
        {  
            cout << "Error creating cuda context";  
            return ;  
        }
        unsigned int freeS, totalS;
        cuMemGetInfo(&freeS, &totalS);
        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
        if (memsize < freeS)
            printf ("Memory allocation size is ok. Require: %d\n", memsize);
        else{
            printf ("Not enough video memory! Return. Require: d\n", memsize);
            cuCtxDetach( cudaContext );
            return;
        }
#endif
        //
        // create the output tensor field
        // TODO: consider hardware limit, do batched operation
        // size of 3x3 matrices per point
        // ATTENTION: sm_11 only support int atomicAdd
        size_t sizeField = numGrids*3*sizeof(float3);

        // copy it to device
        float3 *d_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeField) );
        cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeField) );

        //
        // port points as float3
        size_t sizePoints = numPoints*sizeof(float3);
        // copy it to device
        float3 * d_points;
        cutilSafeCall( cudaMalloc((void **)&d_points, sizePoints) );
        cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice,0) );

        size_t sizeLog = numPoints*sizeof(int2);
        int2 * d_log; // [id, as_voter_#votees, as_votee_#voters]
        cutilSafeCall( cudaMalloc((void **)&d_log, sizeLog) );
        cutilSafeCall( cudaMemset(d_log, 0, sizeLog) );
//        cutilSafeCall( cudaMemcpyAsync(d_log, h_log, sizeLog, cudaMemcpyHostToDevice, 0) );

        size_t sizeSparseStick = numPoints*sizeof(float4);
        float4 * d_sparsestick;
        cutilSafeCall( cudaMalloc((void **)&d_sparsestick, sizeSparseStick) );
        cutilSafeCall( cudaMemcpy(d_sparsestick, h_sparsestick, sizeSparseStick, cudaMemcpyHostToDevice) );

        size_t sizeGrid = numGrids*sizeof(float3);
        float3 *d_grids;
        cutilSafeCall( cudaMalloc((void **)&d_grids, sizeGrid) );
        cutilSafeCall( cudaMemcpy(d_grids, h_grids, sizeGrid, cudaMemcpyHostToDevice) );
        

    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event0, stop_event0) );
    printf("EVENT timed time before launching kernel dense stick voting:\t%.2f \n", elapsed_time );


        dim3 dimGrid( (numGrids - 1)/BLOCK_DIM + 1, BLOCK_DIM);
        printf ("[GPU] deisgned size: %d\n", (numGrids-1)/BLOCK_DIM + 1);
    cudaEventRecord(start_event, 0);
        dense_grid_stick_voting_kernel <<< (numGrids-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_sparsestick, d_grids, d_points, sigma, numGrids, numPoints, d_log);
        getLastCudaError("dense_grid_stick_voting_kernel() execution failed\n");
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("EVENT timed time of kernel dense grid stick voting:\t%.2f \n", elapsed_time );

        // output
        cutilSafeCall( cudaMemcpy(h_fieldarray, d_fieldarray, sizeField, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(h_log, d_log, sizeLog, cudaMemcpyDeviceToHost) );

        //clean up
        cutilSafeCall( cudaFree(d_fieldarray) );
        cutilSafeCall( cudaFree(d_sparsestick) );
        cutilSafeCall( cudaFree(d_points) );
        cutilSafeCall( cudaFree(d_grids) );
        cutilSafeCall( cudaFree(d_log) ); // TODO: remove the log variables
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time0, start_event0, stop_event0) );
    printf("EVENT timed time everything in densestickvoting:\t%.2f \n\n", elapsed_time0 );
#ifdef check_memory_size
        cuCtxDetach( cudaContext );
#endif
        cutilSafeCall( cudaDeviceReset() );
    }


    // @h_fieldarray : output init status of outputing field
    // @h_points : input coords of points
    // @h_sparsestick: input sparse stick, eigen values and eigen vectors
    // @sigma : input sigma size
    // @numPoints : input number of points
    // @h_logg : output log
    extern "C"
    void CudaVoting::denseStickVoting(float3 * h_fieldarray,
                const float3 * h_points, const float4 * h_sparsestick,
                const float sigma, const int numPoints, int2 * h_log)
    {
    cudaEvent_t start_event, start_event0, stop_event, stop_event0;
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );
    cutilSafeCall( cudaEventCreate(&start_event0) );
    cutilSafeCall( cudaEventCreate(&stop_event0) );
    float elapsed_time0, elapsed_time;
    cudaEventRecord(start_event0, 0);

        // select device and initialize CUDA environment
        cuInit(0);
//        routineCheck();
#ifdef check_memory_size
        // check whether the field size to be created fit in mem
        size_t memsize = sizeof(float)*4*4*numPoints; //(3x3 + 3x1 = 3x4) + 4x1
        CUdevice cudaDevice;  
        CUresult result = cuDeviceGet(&cudaDevice, 0);  
        if(result!= CUDA_SUCCESS)  
        {  
            cout << "Error fetching cuda device";  
            return ;  
        }  
        CUcontext cudaContext;  
        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
        if(result != CUDA_SUCCESS)  
        {  
            cout << "Error creating cuda context";  
            return ;  
        }
        unsigned int freeS, totalS;
        cuMemGetInfo(&freeS, &totalS);
        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
        if (memsize < freeS)
            printf ("Memory allocation size is ok. Require: %d\n", memsize);
        else{
            printf ("Not enough video memory! Return. Require: d\n", memsize);
            cuCtxDetach( cudaContext );
            return;
        }
#endif
        //
        // create the output tensor field
        // TODO: consider hardware limit, do batched operation
        // size of 3x3 matrices per point
        // ATTENTION: sm_11 only support int atomicAdd
        size_t sizeField = numPoints*3*sizeof(float3);

        // copy it to device
        float3 *d_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeField) );
        //cutilSafeCall( cudaMemcpyAsync(d_fieldarray, h_fieldarray, sizeField, cudaMemcpyHostToDevice,0) );
        // Don't need to copy field, since it should be 0 everywhere
        //cutilSafeCall( cudaMemcpy(d_fieldarray, h_fieldarray, sizeField, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeField) );

        //
        // port points as float3
        size_t sizePoints = numPoints*sizeof(float3);
        // copy it to device
        float3 * d_points;
        cutilSafeCall( cudaMalloc((void **)&d_points, sizePoints) );
        //cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice,0) );
        cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice) );

        size_t sizeLog = numPoints*sizeof(int2);
        int2 * d_log; // [id, as_voter_#votees, as_votee_#voters]
        cutilSafeCall( cudaMalloc((void **)&d_log, sizeLog) );
        cutilSafeCall( cudaMemset(d_log, 0, sizeLog) );
//        cutilSafeCall( cudaMemcpyAsync(d_log, h_log, sizeLog, cudaMemcpyHostToDevice, 0) );

        size_t sizeSparseStick = numPoints*sizeof(float4);
        float4 * d_sparsestick;
        cutilSafeCall( cudaMalloc((void **)&d_sparsestick, sizeSparseStick) );
        cutilSafeCall( cudaMemcpy(d_sparsestick, h_sparsestick, sizeSparseStick, cudaMemcpyHostToDevice) );

    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event0, stop_event0) );
    printf("EVENT timed time before launching kernel dense stick voting:\t%.2f \n", elapsed_time );


        dim3 dimGrid( (numPoints - 1)/BLOCK_DIM + 1, BLOCK_DIM);
        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
    cudaEventRecord(start_event, 0);
        dense_stick_voting_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_sparsestick, d_points, sigma, numPoints, d_log);
        getLastCudaError("dense_stick_voting_kernel() execution failed\n");
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("EVENT timed time of kernel dense stick voting:\t%.2f \n", elapsed_time );

        // output
        cutilSafeCall( cudaMemcpy(h_fieldarray, d_fieldarray, sizeField, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(h_log, d_log, sizeLog, cudaMemcpyDeviceToHost) );

        //clean up
        cutilSafeCall( cudaFree(d_fieldarray) );
        cutilSafeCall( cudaFree(d_sparsestick) );
        cutilSafeCall( cudaFree(d_points) );
        cutilSafeCall( cudaFree(d_log) ); // TODO: remove the log variables
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time0, start_event0, stop_event0) );
    printf("EVENT timed time everything in densestickvoting:\t%.2f \n\n", elapsed_time0 );
#ifdef check_memory_size
        cuCtxDetach( cudaContext );
#endif
        cutilSafeCall( cudaDeviceReset() );
    }

    // @h_fieldarray : output init status of outputing field
    // @h_points : input coords of points
    // @h_sparsestick: input sparse stick, eigen values and eigen vectors
    // @sigma : input sigma size
    // @numPoints : input number of points
    // @h_logg : output log
    extern "C"
    void CudaVoting::sparseBallVoting(float3 * h_fieldarray,
                const float3 * h_points,
                const float sigma, const int numPoints, int2 * h_log)
    {
    cudaEvent_t start_event, start_event0, stop_event, stop_event0;
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );
    cutilSafeCall( cudaEventCreate(&start_event0) );
    cutilSafeCall( cudaEventCreate(&stop_event0) );
    float elapsed_time0, elapsed_time;
    cudaEventRecord(start_event0, 0);

        // select device and initialize CUDA environment
        cuInit(0);
//        routineCheck();
#ifdef check_memory_size
        // check whether the field size to be created fit in mem
        size_t memsize = sizeof(float)*3*4*numPoints; //(3x3 + 3x1 = 3x4)
        CUdevice cudaDevice;  
        CUresult result = cuDeviceGet(&cudaDevice, 0);  
        if(result!= CUDA_SUCCESS)  
        {  
            cout << "Error fetching cuda device";  
            return ;  
        }  
        CUcontext cudaContext;  
        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
        if(result != CUDA_SUCCESS)  
        {  
            cout << "Error creating cuda context";  
            return ;  
        }
        unsigned int freeS, totalS;
        cuMemGetInfo(&freeS, &totalS);
        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
        if (memsize < freeS)
            printf ("Memory allocation size is ok. Require: %d\n", memsize);
        else{
            printf ("Not enough video memory! Return. Require: d\n", memsize);
            cuCtxDetach( cudaContext );
            return;
        }
#endif
        //
        // create the output tensor field
        // TODO: consider hardware limit, do batched operation
        // size of 3x3 matrices per point
        // ATTENTION: sm_11 only support int atomicAdd
        size_t sizeField = numPoints*3*sizeof(float3);

        // copy it to device
        float3 *d_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeField) );
        //cutilSafeCall( cudaMemcpyAsync(d_fieldarray, h_fieldarray, sizeField, cudaMemcpyHostToDevice,0) );
        //cutilSafeCall( cudaMemcpyAsync(d_fieldarray, h_fieldarray, sizeField, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeField) );

        //
        // port points as float3
        size_t sizePoints = numPoints*sizeof(float3);
        // copy it to device
        float3 * d_points;
        cutilSafeCall( cudaMalloc((void **)&d_points, sizePoints) );
        //cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice,0) );
        cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice) );

        size_t sizeLog = numPoints*sizeof(int2);
        int2 * d_log; // [id, as_voter_#votees, as_votee_#voters]
        cutilSafeCall( cudaMalloc((void **)&d_log, sizeLog) );
        cutilSafeCall( cudaMemset(d_log, 0, sizeLog) );
//        cutilSafeCall( cudaMemcpyAsync(d_log, h_log, sizeLog, cudaMemcpyHostToDevice, 0) );

    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event0, stop_event0) );
    printf("EVENT timed time before launching kernel :\t%.2f \n", elapsed_time );

        dim3 dimGrid( (numPoints - 1)/BLOCK_DIM + 1, BLOCK_DIM);
        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
    cudaEventRecord(start_event, 0);
        sparse_ball_voting_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_points, sigma, numPoints, d_log);
        getLastCudaError("sparse_ball_voting_kernel() execution failed\n");
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("EVENT timed time of kernel 1:\t%.2f \n", elapsed_time );

        // output
        cutilSafeCall( cudaMemcpy(h_fieldarray, d_fieldarray, sizeField, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(h_log, d_log, sizeLog, cudaMemcpyDeviceToHost) );

        //clean up
        cutilSafeCall( cudaFree(d_fieldarray) );
        cutilSafeCall( cudaFree(d_points) );
        cutilSafeCall( cudaFree(d_log) );
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time0, start_event0, stop_event0) );
    printf("EVENT timed time everything in sparseballvoting:\t%.2f \n\n", elapsed_time0 );
#ifdef check_memory_size
        cuCtxDetach( cudaContext );
#endif
        cutilSafeCall( cudaDeviceReset() );
    }


    //*************************************
    // Real dense grid voting
    //*************************************

    __global__ void real_pointgrid_creation(float3 * d_pointgrid, 
                const float3 * d_points, const int hw_size, const float cell_size,
                const int numPoints, 
                const float min_x,
                const float min_y,
                const float min_z,
                int2 * d_log)
    {
        int token = threadIdx.x+blockIdx.x*blockDim.x; // over all points
        if (token >= numPoints) return;

        float3 point = d_points[token]; // base point
        int idx = round( (point.x-min_x) / cell_size)+hw_size; // added correct or margins
        int idy = round( (point.y-min_y) / cell_size)+hw_size;
        int idz = round( (point.z-min_z) / cell_size)+hw_size;

        // update d_pointgrid
        d_pointgrid[token] = make_float3(idx, idy, idz);
    }
    
//    __global__ void real_dense_stick_grid_voting(float3 * field, const float3 * d_pointgrid, 
//                const float4 * sparsestick, 
//                const int hw_size, const float cell_size,
//                const float sigma, const int numPoints, 
//                const unsigned int maxGridSize, 
//                const unsigned int grid_dimx,
//                const unsigned int grid_dimy,
//                const unsigned int grid_dimz,
//                const float min_x,
//                const float min_y,
//                const float min_z,
//                int2 * logg)
//    {
//
//        __shared__ float3 tmpfield[BLOCK_DIM*3];
//        tmpfield[threadIdx.x*3+0] = make_float3(0,0,0);
//        tmpfield[threadIdx.x*3+1] = make_float3(0,0,0);
//        tmpfield[threadIdx.x*3+2] = make_float3(0,0,0);
//        __syncthreads();
//
//        int token = threadIdx.x+blockIdx.x*blockDim.x; // over all points
//        if (token >= maxGridSize) return;
//
//        // votee P : position cell coordinate
//        int Pz = token / (grid_dimx*grid_dimy);
//        int Pxy = token % (grid_dimx*grid_dimy);
//        int Px = Pxy % grid_dimx;
//        int Py = Pxy / grid_dimx; // coord origined locally
//
//        // global coord
//        float3 P = make_float3((float)Px, (float)Py, (float)Pz );
//
//        #pragma unroll 64
//        for(unsigned int voter_i = 0; voter_i < numPoints; voter_i++)
//        {
//            float3 O_int = d_pointgrid[voter_i];
//
//            // test O_int : PASS
// //           logg[voter_i].x = O_int.x;
// //           logg[voter_i].y = O_int.y;
//
//            float3 O = make_float3((float)O_int.x, (float)O_int.y, (float)O_int.z);
//
//            if ( abs(Px - O_int.x) > hw_size || abs(Py - O_int.y) > hw_size || abs(Pz - O_int.z) > hw_size)
//              continue;
//            
//            float3 v = (P - O)*cell_size;
//            float l = length(v);
//            float scaled_dist = __fdividef(l,sigma);//sqrt(l)/sigma;
//            if(l == 0 || scaled_dist >=3)
//              continue;
//
//            float4 _sparseStick = sparsestick[voter_i];
//            float stick_saliency = _sparseStick.x;
//            float3 vn = make_float3(_sparseStick.y, _sparseStick.z, _sparseStick.w);
//            float _norm = length(vn);
//            if (_norm > 0) 
//                vn = make_float3( __fdividef(vn.x, _norm), __fdividef(vn.y, _norm), __fdividef(vn.z, _norm));
//
//            v = make_float3( __fdividef(v.x, l), __fdividef(v.y, l), __fdividef(v.z, l));
//            float vvn = __fmul_rn(v.x, vn.x) + __fmul_rn(v.y, vn.y) + __fmul_rn(v.z, vn.z); //dot(v, vn);
//            if (vvn < 0) vn = -vn;
//            float theta = asin(vvn);
//
//            if (fabs(theta) <= M_PI/4 || fabs(theta) >=3*M_PI/4)
//            {
//                
//                float3 crossvvn = make_float3(__fmul_rn(v.y, vn.z) - __fmul_rn(v.z, vn.y),
//                                              __fmul_rn(v.z, vn.x) - __fmul_rn(v.x, vn.z),
//                                              __fmul_rn(v.x, vn.y) - __fmul_rn(v.y, vn.x)
//                                              );
//                float3 vt= make_float3(__fmul_rn(vn.y, crossvvn.z) - __fmul_rn(vn.z, crossvvn.y),
//                                __fmul_rn(vn.z, crossvvn.x) - __fmul_rn(vn.x, crossvvn.z),
//                                __fmul_rn(vn.x, crossvvn.y) - __fmul_rn(vn.y, crossvvn.x)
//                                );
//                float vtl = length(vt);
//                if (vtl > 0)
//                    vt = make_float3( __fdividef(vt.x, vtl), __fdividef(vt.y, vtl), __fdividef(vt.z, vtl));
//                float vvt = __fmul_rn(v.x, vt.x) + __fmul_rn(v.y, vt.y) + __fmul_rn(v.z, vt.z);
//                if (vvt < 0) vt = -vt;
//
//                float3 vc = product(vn, __cosf(2*theta)) - product(vt, __sinf(2*theta));
//                float c = pow(scaled_dist, 2)*pow(scaled_dist-3, 4)*(1.0/16.0); // smooth version
//                float r = length(vc);
//                float DF=exp( -(4*r*r*theta*theta + c/(r*r)) * (1.0/(sigma*sigma)) ) * pow(__cosf(theta), 8);
//                float3 Sp[3];
//                Sp[0] = product(product(make_float3(vc.x*vc.x, vc.x*vc.y, vc.x*vc.z), DF), stick_saliency);
//                Sp[1] = product(product(make_float3(vc.y*vc.x, vc.y*vc.y, vc.y*vc.z), DF), stick_saliency);
//                Sp[2] = product(product(make_float3(vc.z*vc.x, vc.z*vc.y, vc.z*vc.z), DF), stick_saliency);
//                tmpfield[threadIdx.x*3 + 0] = Sp[0] + tmpfield[threadIdx.x*3 + 0] ;
//                tmpfield[threadIdx.x*3 + 1] = Sp[1] + tmpfield[threadIdx.x*3 + 1];
//                tmpfield[threadIdx.x*3 + 2] = Sp[2] + tmpfield[threadIdx.x*3 + 2];
//            } // end of angle condition
//        }// end of for all voters
//        __syncthreads();
//        field[token*3 + 0] = tmpfield[threadIdx.x*3 + 0];
//        field[token*3 + 1] = tmpfield[threadIdx.x*3 + 1];
//        field[token*3 + 2] = tmpfield[threadIdx.x*3 + 2];
//    }


//Depracated
//    // @h_fieldarray : output init status of outputing field
//    // @h_densegrid: output dense grid setups (coordinates); used for further visualization or registration
//    // @h_points : input coords of points
//    // @h_sparsestick: input sparse stick, eigen values and eigen vectors
//    // @maxGridSize: input maximum possible number of cells in Grid
//    // @sigma : input sigma size
//    // @cell_size: input cell size of grid
//    // @hw_size : input half_window size <- number of cells needs to be extended per point
//    // @numPoints : input number of points
//    // @min_x(yz): min coord in meter (origin of the grid)
//    // @h_logg : output log
//    extern "C"
//    void CudaVoting::realDenseGridStickVoting(float3 * h_fieldarray,
//                const float3 * h_points, const float4 * h_sparsestick, 
//                const size_t maxGridSize, const int hw_size,
//                const float sigma, const float cell_size, const int numPoints, 
//                const unsigned int grid_dimx,
//                const unsigned int grid_dimy,
//                const unsigned int grid_dimz,
//                const float min_x,
//                const float min_y,
//                const float min_z,
//                int2 * hh_log)
//    {
//        // events for timing 
//        cudaEvent_t start_event, start_event0, stop_event, stop_event0;
//        cutilSafeCall( cudaEventCreate(&start_event) );
//        cutilSafeCall( cudaEventCreate(&stop_event) );
//        cutilSafeCall( cudaEventCreate(&start_event0) );
//        cutilSafeCall( cudaEventCreate(&stop_event0) );
//        float elapsed_time0, elapsed_time;
//        cudaEventRecord(start_event0, 0);
//
//        // select device and initialize CUDA environment
//        cuInit(0);
//#ifdef check_memory_size
//        // check whether the field size to be created fit in mem
//        size_t memsize = sizeof(float)*3*4*numPoints; //(3x3 + 3x1 = 3x4)
//        CUdevice cudaDevice;  
//        CUresult result = cuDeviceGet(&cudaDevice, 0);  
//        if(result!= CUDA_SUCCESS)  
//        {  
//            cout << "Error fetching cuda device";  
//            return ;  
//        }  
//        CUcontext cudaContext;  
//        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
//        if(result != CUDA_SUCCESS)  
//        {  
//            cout << "Error creating cuda context";  
//            return ;  
//        }
//        unsigned int freeS, totalS;
//        cuMemGetInfo(&freeS, &totalS);
//        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
//        if (memsize < freeS)
//          printf ("Memory allocation size is ok. Require: %d\n", memsize);
//        else{
//            printf ("Not enough video memory! Return. Require: d\n", memsize);
//            cuCtxDetach( cudaContext );
//            return;
//        }
//#endif
//        //
//        // create the output tensor field
//        // TODO: consider hardware limit, do batched operation
//        // size of 3x3 matrices per point
//        // ATTENTION: sm_11 only support int atomicAdd
//        size_t sizeField = maxGridSize*3*sizeof(float3);
//
//        // copy it to device
//        float3 *d_fieldarray;
//        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeField) );
//        cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeField) );
//
//        //
//        // port points as float3
//        size_t sizePoints = numPoints*sizeof(float3);
//        float3 * d_points;
//        cutilSafeCall( cudaMalloc((void **)&d_points, sizePoints) );
//        cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice, 0) );
//
//        size_t sizeLog = numPoints*sizeof(int2);
//        int2 * d_log; // [id, as_voter_#votees, as_votee_#voters]
//        cutilSafeCall( cudaMalloc((void **)&d_log, sizeLog) );
//        cutilSafeCall( cudaMemset(d_log, 0, sizeLog) );
//
//        size_t sizePointGrid = numPoints*sizeof(float3);
//        float3 * d_pointgrid; // kept on GPU
//        cutilSafeCall( cudaMalloc((void **)&d_pointgrid, sizePointGrid) );
//        cutilSafeCall( cudaMemset(d_pointgrid, 0, sizePointGrid) );
//
//        size_t sizeSparseStick = numPoints*sizeof(float4);
//        float4 * d_sparsestick;
//        cutilSafeCall( cudaMalloc((void **)&d_sparsestick, sizeSparseStick) );
//        cutilSafeCall( cudaMemcpy(d_sparsestick, h_sparsestick, sizeSparseStick, cudaMemcpyHostToDevice) );
//
//        cudaEventRecord(stop_event0, 0);
//        cudaEventSynchronize(stop_event0);
//        cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event0, stop_event0) );
//        printf("EVENT timed time before launching kernel :\t%.2f \n", elapsed_time );
//
//        // =========================================
//        // KERNEL1:
//
//        // create grid for the points; get the number of cells needs to be created
//        // create d_pointgrid, saving only the nearest point
//        // d_pointgrid: [{int_x, int_y, int_z}]
//        printf ("[GPU] deisgned size for kernel d_pointgrid: %d\n", (numPoints-1)/BLOCK_DIM_G + 1);
//
//        cudaEventRecord(start_event, 0);
//        real_pointgrid_creation <<< (numPoints-1)/BLOCK_DIM_G + 1, BLOCK_DIM_G >>> (d_pointgrid, d_points, 
//                    hw_size, cell_size, numPoints, 
//                    min_x, min_y, min_z, d_log);
//        getLastCudaError("real_dense_grid_creation() execution failed\n");
//        cudaEventRecord(stop_event, 0);
//
//        cudaEventSynchronize(stop_event);
//        cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
//        printf("EVENT timed time of kernel d_pointgrid:\t%.2f \n", elapsed_time );
//
//        //----------------------------------------
//        // KERNEL2:
//        // do dense voting based on the grids created
//        printf ("[GPU] deisgned size for kernel real_dense_grid_voting(): %ld\n", (maxGridSize-1)/BLOCK_DIM + 1);
//
//        cudaEventRecord(start_event, 0);
//        real_dense_stick_grid_voting <<< (maxGridSize-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_pointgrid, d_sparsestick, 
//                                                                                hw_size, cell_size, sigma, numPoints, maxGridSize, 
//                                                                                grid_dimx, grid_dimy, grid_dimz,
//                                                                                min_x, min_y, min_z, d_log);
//        getLastCudaError("real_dense_grid_voting() execution failed\n");
//        cudaEventRecord(stop_event, 0);
//
//        cudaEventSynchronize(stop_event);
//        cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
//        printf("EVENT timed time of kernel real_dense_grid_voting()2:\t%.2f \n", elapsed_time );
//        //=========================================
//
//        // output
//        cutilSafeCall( cudaMemcpy(h_fieldarray, d_fieldarray, sizeField, cudaMemcpyDeviceToHost) );
//        cutilSafeCall( cudaMemcpy(hh_log, d_log, sizeLog, cudaMemcpyDeviceToHost) );
//
//        //clean up
//        cutilSafeCall( cudaFree(d_fieldarray) );
//        cutilSafeCall( cudaFree(d_points) );
//        cutilSafeCall( cudaFree(d_sparsestick) );
//        cutilSafeCall( cudaFree(d_log) );
//        cudaEventRecord(stop_event0, 0);
//        cudaEventSynchronize(stop_event0);
//        cutilSafeCall( cudaEventElapsedTime(&elapsed_time0, start_event0, stop_event0) );
//        printf("EVENT timed time everything in dense creation:\t%.2f \n\n", elapsed_time0 );
//#ifdef check_memory_size
//        cuCtxDetach( cudaContext );
//#endif
//        cutilSafeCall( cudaDeviceReset() );
//
//    }

    // @h_fieldarray : inputing field, comprimized of 33 matrices
    // @stick: output, stick saliency
    // @plate: output, plate saliency
    // @ball: output, ball saliency
    // @numPoints : input number of points
    extern "C"
    void CudaVoting::tensorSplit(
                    float3 * h_fieldarray, 
                    float * stick,
                    float * plate,
                    float * ball,
                    const int numPoints
                    )
    {
    cudaEvent_t start_event, start_event0, stop_event, stop_event0;
    cutilSafeCall( cudaEventCreate(&start_event) );
    cutilSafeCall( cudaEventCreate(&stop_event) );
    cutilSafeCall( cudaEventCreate(&start_event0) );
    cutilSafeCall( cudaEventCreate(&stop_event0) );
    float elapsed_time0, elapsed_time;
    cudaEventRecord(start_event0, 0);

        // select device and initialize CUDA environment
        cuInit(0);
//        routineCheck();
#ifdef check_memory_size
        // check whether the field size to be created fit in mem
        size_t memsize = sizeof(float)*3*4*numPoints; //(3x3 + 3x1 = 3x4)
        CUdevice cudaDevice;  
        CUresult result = cuDeviceGet(&cudaDevice, 0);  
        if(result!= CUDA_SUCCESS)  
        {  
            cout << "Error fetching cuda device";  
            return ;  
        }  
        CUcontext cudaContext;  
        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
        if(result != CUDA_SUCCESS)  
        {  
            cout << "Error creating cuda context";  
            return ;  
        }
        unsigned int freeS, totalS;
        cuMemGetInfo(&freeS, &totalS);
        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
        if (memsize < freeS)
            printf ("Memory allocation size is ok. Require: %d\n", memsize);
        else{
            printf ("Not enough video memory! Return. Require: d\n", memsize);
            cuCtxDetach( cudaContext );
            return;
        }
#endif
        //
        // Start working
        //
        size_t sizeField = numPoints*3*sizeof(float3);

        // copy it to device
        float3 *d_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeField) );
        cutilSafeCall( cudaMemcpyAsync(d_fieldarray, h_fieldarray, sizeField, cudaMemcpyHostToDevice,0) );
        //cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeField) );

        float * d_stick, * d_plate, * d_ball;
        size_t sizeSaliency = numPoints*sizeof(float);
        cutilSafeCall( cudaMalloc((void **)&d_stick, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_plate, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_ball, sizeSaliency) );

    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event0, stop_event0) );
    printf("EVENT timed time before launching kernel :\t%.2f \n", elapsed_time );

        dim3 dimGrid( (numPoints - 1)/BLOCK_DIM + 1, BLOCK_DIM);
        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
    cudaEventRecord(start_event, 0);
        tensor_split_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_stick, d_plate, d_ball, numPoints);
        getLastCudaError("tensor_split_kernel() execution failed\n");
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("EVENT timed time of kernel tensor_split_kernel():\t%.2f \n", elapsed_time );

        // output
        cutilSafeCall( cudaMemcpy(stick, d_stick, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(plate, d_plate, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(ball, d_ball, sizeSaliency, cudaMemcpyDeviceToHost) );

        //clean up
        cutilSafeCall( cudaFree(d_fieldarray) );
        cutilSafeCall( cudaFree(d_stick) );
        cutilSafeCall( cudaFree(d_plate) );
        cutilSafeCall( cudaFree(d_ball) );
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time0, start_event0, stop_event0) );
    printf("EVENT timed time everything in tensor_split:\t%.2f \n\n", elapsed_time0 );
#ifdef check_memory_size
        cuCtxDetach( cudaContext );
#endif
        cutilSafeCall( cudaDeviceReset() );
    }


    extern "C"
    void CudaVoting::tensorSplitWithField(
                    float3 * h_fieldarray, 
                    float * stick,
                    float * plate,
                    float * ball,
                    float3 * h_stick_field,
                    float3 * h_plate_field,
                    const int numPoints
                    )
    {
        cudaEvent_t start_event, start_event0, stop_event, stop_event0;
        cutilSafeCall( cudaEventCreate(&start_event) );
        cutilSafeCall( cudaEventCreate(&stop_event) );
        cutilSafeCall( cudaEventCreate(&start_event0) );
        cutilSafeCall( cudaEventCreate(&stop_event0) );
        float elapsed_time0, elapsed_time;
        cudaEventRecord(start_event0, 0);

        // select device and initialize CUDA environment
        cuInit(0);
        //        routineCheck();
#ifdef check_memory_size
        // check whether the field size to be created fit in mem
        size_t memsize = sizeof(float)*3*4*numPoints; //(3x3 + 3x1 = 3x4)
        CUdevice cudaDevice;  
        CUresult result = cuDeviceGet(&cudaDevice, 0);  
        if(result!= CUDA_SUCCESS)  
        {  
            cout << "Error fetching cuda device";  
            return ;  
        }  
        CUcontext cudaContext;  
        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
        if(result != CUDA_SUCCESS)  
        {  
            cout << "Error creating cuda context";  
            return ;  
        }
        unsigned int freeS, totalS;
        cuMemGetInfo(&freeS, &totalS);
        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
        if (memsize < freeS)
          printf ("Memory allocation size is ok. Require: %d\n", memsize);
        else{
            printf ("Not enough video memory! Return. Require: d\n", memsize);
            cuCtxDetach( cudaContext );
            return;
        }
#endif
        //
        // Start working
        //
        size_t sizeField = numPoints*3*sizeof(float3);

        // copy it to device
        float3 *d_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeField) );
        cutilSafeCall( cudaMemcpyAsync(d_fieldarray, h_fieldarray, sizeField, cudaMemcpyHostToDevice,0) );
        //cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeField) );

        float * d_stick, * d_plate, * d_ball;
        size_t sizeSaliency = numPoints*sizeof(float);
        cutilSafeCall( cudaMalloc((void **)&d_stick, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_plate, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_ball, sizeSaliency) );

        float3 * d_stick_field, * d_plate_field;
        size_t sizeSparseEV = numPoints*sizeof(float3);
        cutilSafeCall( cudaMalloc((void **)&d_stick_field, sizeSparseEV) );
        cutilSafeCall( cudaMalloc((void **)&d_plate_field, sizeSparseEV) );

        cudaEventRecord(stop_event0, 0);
        cudaEventSynchronize(stop_event0);
        cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event0, stop_event0) );
        printf("EVENT timed time before launching kernel :\t%.2f \n", elapsed_time );

        dim3 dimGrid( (numPoints - 1)/BLOCK_DIM + 1, BLOCK_DIM);
        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
        cudaEventRecord(start_event, 0);
        //tensor_split_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_stick, d_plate, d_ball, numPoints);
        tensor_split_with_eigenvectors_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, d_stick, d_plate, d_ball,
                    d_stick_field, d_plate_field, numPoints);
        getLastCudaError("tensor_split_kernel() execution failed\n");
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
        printf("EVENT timed time of kernel tensor_split_kernel():\t%.2f \n", elapsed_time );

        // output
        cutilSafeCall( cudaMemcpy(stick, d_stick, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(plate, d_plate, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(h_stick_field, d_stick_field, sizeSparseEV, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(h_plate_field, d_plate_field, sizeSparseEV, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(ball, d_ball, sizeSaliency, cudaMemcpyDeviceToHost) );

        //clean up
        cutilSafeCall( cudaFree(d_fieldarray) );
        cutilSafeCall( cudaFree(d_stick) );
        cutilSafeCall( cudaFree(d_plate) );
        cutilSafeCall( cudaFree(d_stick_field) );
        cutilSafeCall( cudaFree(d_plate_field) );
        cutilSafeCall( cudaFree(d_ball) );
        cudaEventRecord(stop_event0, 0);
        cudaEventSynchronize(stop_event0);
        cutilSafeCall( cudaEventElapsedTime(&elapsed_time0, start_event0, stop_event0) );
        printf("EVENT timed time everything in tensor_split:\t%.2f \n\n", elapsed_time0 );
#ifdef check_memory_size
        cuCtxDetach( cudaContext );
#endif
        cutilSafeCall( cudaDeviceReset() );


    }

    // @h_points : input coords of points
    // @h_sparsestick: input sparse stick, eigen values and eigen vectors
    // @maxGridSize: input maximum possible number of cells in Grid
    // @sigma : input sigma size
    // @cell_size: input cell size of grid
    // @hw_size : input half_window size <- number of cells needs to be extended per point
    // @numPoints : input number of points
    // @min_x(yz): min coord in meter (origin of the grid)
    // @h_logg : output log
    extern "C"
    void CudaVoting::unifiedDenseGridStickVoting(
                const float3 * h_points, 
                const size_t maxGridSize, const int hw_size,
                const float sigma, const float cell_size, const int numPoints, 
                const unsigned int grid_dimx,
                const unsigned int grid_dimy,
                const unsigned int grid_dimz,
                const float min_x,
                const float min_y,
                const float min_z,
                float * stick_saliency_grid,
                float * plate_saliency_grid,
                float * ball_saliency_grid,
                float3 * grid_normals,
                int2 * hh_log)
    {
        // events for timing 
        cudaEvent_t start_event, start_event0, stop_event, stop_event0;
        cutilSafeCall( cudaEventCreate(&start_event) );
        cutilSafeCall( cudaEventCreate(&stop_event) );
        cutilSafeCall( cudaEventCreate(&start_event0) );
        cutilSafeCall( cudaEventCreate(&stop_event0) );
        float elapsed_time0, elapsed_time;
        cudaEventRecord(start_event0, 0);

        // select device and initialize CUDA environment
        cuInit(0);
#ifdef check_memory_size
        // check whether the field size to be created fit in mem
        size_t memsize0 = sizeof(float3)*numPoints; //points
        size_t memsize1 = sizeof(float3)*3*numPoints; //sparse field
        size_t memsize2 = sizeof(int2)*numPoints; //log
        size_t memsize3 = 2*numPoints*sizeof(float3); // sparse tensors
        size_t memsize4 = 3*numPoints*sizeof(float); // sparse saliency
        size_t memsize5 = sizeof(float3)*numPoints; // point grids
        size_t memsize6 = sizeof(float3)*maxGridSize; // Grid size
        size_t memsize7 = sizeof(float3)*3*maxGridSize; // dense field size
        size_t memsize8 = 2*maxGridSize*sizeof(float3); // dense tensors
        size_t memsize9 = 3*maxGridSize*sizeof(float); // dense saliency

        size_t memsize = memsize0 + memsize1 + memsize2 + memsize3 + memsize4 + memsize5 + memsize6 + memsize7 + memsize8 + memsize9;

        cout << "unified voting Estimated memory usage: " << endl;
        cout << "----------------------------------------" << endl;
        cout << "raw points " << memsize0 << endl;
        cout << "sparse field " << memsize1 << endl;
        cout << "log " << memsize2 << endl;
        cout << "sparse tensors " << memsize3 << endl;
        cout << "sparse saliency " << memsize4 << endl;
        cout << "point grid " << memsize5 << endl;
        cout << "dense grid " << memsize6 << endl;
        cout << "dense field  " << memsize7 << endl;
        cout << "dense tensors " << memsize8 << endl;
        cout << "dense saliency " << memsize9 << endl;
        cout << "----------------------------------------" << endl;
        cout << "Total: " << memsize << endl;


        CUdevice cudaDevice;  
        CUresult result = cuDeviceGet(&cudaDevice, 0);  
        if(result!= CUDA_SUCCESS)  
        {  
            cout << "Error fetching cuda device";  
            return ;  
        }  
        CUcontext cudaContext;  
        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
        if(result != CUDA_SUCCESS)  
        {  
            cout << "Error creating cuda context";  
            return ;  
        }
        size_t freeS, totalS;
        cuMemGetInfo(&freeS, &totalS);
        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
        if (memsize < freeS)
          printf ("Memory allocation size is ok. Require: %d\n", memsize);
        else{
            printf ("Not enough video memory! Return. Require: d\n", memsize);
            cuCtxDetach( cudaContext );
            return;
        }
#endif

        // start real work

        //
        // create sparse tensor field
        // size of 3x3 matrices per point
        size_t sizeField = numPoints*3*sizeof(float3);
        // copy it to device
        float3 *ds_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&ds_fieldarray, sizeField) );
        cutilSafeCall( cudaMemset(ds_fieldarray, 0, sizeField) );

        //
        // port points as float3
        size_t sizePoints = numPoints*sizeof(float3);
        float3 * d_points;
        cutilSafeCall( cudaMalloc((void **)&d_points, sizePoints) );
        cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice, 0) );

        size_t sizeLog = numPoints*sizeof(int2);
        int2 * d_log; // [id, as_voter_#votees, as_votee_#voters]
        cutilSafeCall( cudaMalloc((void **)&d_log, sizeLog) );
        cutilSafeCall( cudaMemset(d_log, 0, sizeLog) );


        // ===============================
        // do sparse voting
        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
        sparse_ball_voting_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (ds_fieldarray, d_points, sigma, numPoints, d_log);
        getLastCudaError("sparse_ball_voting_kernel() execution failed\n");
        

        // ===============================
        // do tensor split
        float3 * d_stick_field, * d_plate_field;
        size_t sizeSparseEV = numPoints*sizeof(float3);
        cutilSafeCall( cudaMalloc((void **)&d_stick_field, sizeSparseEV) );
        cutilSafeCall( cudaMalloc((void **)&d_plate_field, sizeSparseEV) );
        float * d_stick, * d_plate, * d_ball;
        size_t sizeSaliency = numPoints*sizeof(float);
        cutilSafeCall( cudaMalloc((void **)&d_stick, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_plate, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_ball, sizeSaliency) );

        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
        tensor_split_with_eigenvectors_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (ds_fieldarray, d_stick, d_plate, d_ball,
                    d_stick_field, d_plate_field, numPoints);
        getLastCudaError("tensor_split_kernel() execution failed\n");


        // ===============================
        // clean un-needed sparse field
        //cutilSafeCall( cudaFree(ds_fieldarray) );

        // ===============================
        // create grid for points
        size_t sizePointGrid = numPoints*sizeof(float3);
        float3 * d_pointgrid; // kept on GPU
        cutilSafeCall( cudaMalloc((void **)&d_pointgrid, sizePointGrid) );
        cutilSafeCall( cudaMemset(d_pointgrid, 0, sizePointGrid) );
        // create grid for the points; get the number of cells needs to be created
        // create d_pointgrid, saving only the nearest point
        // d_pointgrid: [{int_x, int_y, int_z}]
        printf ("[GPU] deisgned size for kernel d_pointgrid: %d\n", (numPoints-1)/BLOCK_DIM_G + 1);
        real_pointgrid_creation <<< (numPoints-1)/BLOCK_DIM_G + 1, BLOCK_DIM_G >>> (d_pointgrid, d_points, 
                    hw_size, cell_size, numPoints, 
                    min_x, min_y, min_z, d_log);
        getLastCudaError("real_dense_grid_creation() execution failed\n");

        //=========================================
        // do dense voting based on the grids created
        size_t sizeGridField = maxGridSize*3*sizeof(float3); // change to output size
        float3 * d_fieldarray;
        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeGridField) );
        cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeGridField) );

        printf ("[GPU] deisgned size for kernel real_dense_grid_voting(): %ld\n", (maxGridSize-1)/BLOCK_DIM + 1);
        real_dense_stick_grid_voting <<< (maxGridSize-1)/BLOCK_DIM + 1, BLOCK_DIM >>> 
            (d_fieldarray, d_pointgrid, d_stick_field, d_stick,
             hw_size, cell_size, sigma, numPoints, maxGridSize, 
             grid_dimx, grid_dimy, grid_dimz,
             min_x, min_y, min_z, d_log);
        getLastCudaError("real_dense_grid_voting() execution failed\n");
        

        // === release memory
        cutilSafeCall( cudaFree(d_stick) );
        cutilSafeCall( cudaFree(d_plate) );
        cutilSafeCall( cudaFree(d_ball) );
        cutilSafeCall( cudaFree(d_points) );
        cutilSafeCall( cudaFree(d_log) );

        cutilSafeCall( cudaFree(ds_fieldarray) );


        //=========================================
        // tensor split on d_fieldarray
        // outputing saliency <- get number of cells in the grid first
        float * dd_stick, * dd_plate, * dd_ball; //on-Device Dense (dd)
        float3 * d_normals; // normal for grid cell
        int numReal = maxGridSize; // TODO: find the real one
        sizeSaliency = numReal*sizeof(float);
        cutilSafeCall( cudaMalloc((void **)&dd_stick, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&dd_plate, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&dd_ball, sizeSaliency) );
        cutilSafeCall( cudaMalloc((void **)&d_normals, sizeSaliency*3) );
        printf ("[GPU] deisgned size for split dense field: %d\n", (numReal-1)/BLOCK_DIM + 1);
        tensor_split_kernel_normal<<< (numReal-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, dd_stick, dd_plate, dd_ball,
                    d_normals, numReal);
        getLastCudaError("tensor_split_kernel_normal() execution failed\n");

        cutilSafeCall( cudaMemcpy(stick_saliency_grid, dd_stick, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(plate_saliency_grid, dd_plate, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(ball_saliency_grid, dd_ball, sizeSaliency, cudaMemcpyDeviceToHost) );
        cutilSafeCall( cudaMemcpy(grid_normals, d_normals, sizeSaliency, cudaMemcpyDeviceToHost) );

        //clean up
        cutilSafeCall( cudaFree(dd_stick) );
        cutilSafeCall( cudaFree(dd_plate) );
        cutilSafeCall( cudaFree(dd_ball) );
        cutilSafeCall( cudaFree(d_fieldarray) );

#ifdef check_memory_size
        cuCtxDetach( cudaContext );
#endif
        cutilSafeCall( cudaDeviceReset() );

    }

//    // @h_points : input coords of points
//    // @h_sparsestick: input sparse stick, eigen values and eigen vectors
//    // @maxGridSize: input maximum possible number of cells in Grid
//    // @sigma : input sigma size
//    // @cell_size: input cell size of grid
//    // @hw_size : input half_window size <- number of cells needs to be extended per point
//    // @numPoints : input number of points
//    // @min_x(yz): min coord in meter (origin of the grid)
//    // @h_logg : output log
//    extern "C"
//    void CudaVoting::unifiedDenseGridPlateVoting(
//                const float3 * h_points, 
//                const size_t maxGridSize, const int hw_size,
//                const float sigma, const float cell_size, const int numPoints, 
//                const unsigned int grid_dimx,
//                const unsigned int grid_dimy,
//                const unsigned int grid_dimz,
//                const float min_x,
//                const float min_y,
//                const float min_z,
//                float * stick_saliency_grid,
//                float * plate_saliency_grid,
//                float * ball_saliency_grid,
//                int2 * hh_log)
//    {
//        // events for timing 
//        cudaEvent_t start_event, start_event0, stop_event, stop_event0;
//        cutilSafeCall( cudaEventCreate(&start_event) );
//        cutilSafeCall( cudaEventCreate(&stop_event) );
//        cutilSafeCall( cudaEventCreate(&start_event0) );
//        cutilSafeCall( cudaEventCreate(&stop_event0) );
//        float elapsed_time0, elapsed_time;
//        cudaEventRecord(start_event0, 0);
//
//        // select device and initialize CUDA environment
//        cuInit(0);
//#ifdef check_memory_size
//        // check whether the field size to be created fit in mem
//        size_t memsize0 = sizeof(float3)*numPoints; //points
//        size_t memsize1 = sizeof(float3)*3*numPoints; //sparse field
//        size_t memsize2 = sizeof(int2)*numPoints; //log
//        size_t memsize3 = 2*numPoints*sizeof(float3); // sparse tensors
//        size_t memsize4 = 3*numPoints*sizeof(float); // sparse saliency
//        size_t memsize5 = sizeof(float3)*numPoints; // point grids
//        size_t memsize6 = sizeof(float3)*maxGridSize; // Grid size
//        size_t memsize7 = sizeof(float3)*3*maxGridSize; // dense field size
//        size_t memsize8 = 2*maxGridSize*sizeof(float3); // dense tensors
//        size_t memsize9 = 3*maxGridSize*sizeof(float); // dense saliency
//
//        size_t memsize = memsize0 + memsize1 + memsize2 + memsize3 + memsize4 + memsize5 + memsize6 + memsize7 + memsize8 + memsize9;
//
//        cout << "unified voting Estimated memory usage: " << endl;
//        cout << "----------------------------------------" << endl;
//        cout << "raw points " << memsize0 << endl;
//        cout << "sparse field " << memsize1 << endl;
//        cout << "log " << memsize2 << endl;
//        cout << "sparse tensors " << memsize3 << endl;
//        cout << "sparse saliency " << memsize4 << endl;
//        cout << "point grid " << memsize5 << endl;
//        cout << "dense grid " << memsize6 << endl;
//        cout << "dense field  " << memsize7 << endl;
//        cout << "dense tensors " << memsize8 << endl;
//        cout << "dense saliency " << memsize9 << endl;
//        cout << "----------------------------------------" << endl;
//        cout << "Total: " << memsize << endl;
//
//
//        CUdevice cudaDevice;  
//        CUresult result = cuDeviceGet(&cudaDevice, 0);  
//        if(result!= CUDA_SUCCESS)  
//        {  
//            cout << "Error fetching cuda device";  
//            return ;  
//        }  
//        CUcontext cudaContext;  
//        result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);  
//        if(result != CUDA_SUCCESS)  
//        {  
//            cout << "Error creating cuda context";  
//            return ;  
//        }
//        size_t freeS, totalS;
//        cuMemGetInfo(&freeS, &totalS);
//        printf("GPU: freeMem = %d , totalMem = %d\n", freeS, totalS);
//        if (memsize < freeS)
//          printf ("Memory allocation size is ok. Require: %d\n", memsize);
//        else{
//            printf ("Not enough video memory! Return. Require: d\n", memsize);
//            cuCtxDetach( cudaContext );
//            return;
//        }
//#endif
//
//        // start real work
//
//        //
//        // create sparse tensor field
//        // size of 3x3 matrices per point
//        size_t sizeField = numPoints*3*sizeof(float3);
//        // copy it to device
//        float3 *ds_fieldarray;
//        cutilSafeCall( cudaMalloc((void **)&ds_fieldarray, sizeField) );
//        cutilSafeCall( cudaMemset(ds_fieldarray, 0, sizeField) );
//
//        //
//        // port points as float3
//        size_t sizePoints = numPoints*sizeof(float3);
//        float3 * d_points;
//        cutilSafeCall( cudaMalloc((void **)&d_points, sizePoints) );
//        cutilSafeCall( cudaMemcpyAsync(d_points, h_points, sizePoints, cudaMemcpyHostToDevice, 0) );
//
//        size_t sizeLog = numPoints*sizeof(int2);
//        int2 * d_log; // [id, as_voter_#votees, as_votee_#voters]
//        cutilSafeCall( cudaMalloc((void **)&d_log, sizeLog) );
//        cutilSafeCall( cudaMemset(d_log, 0, sizeLog) );
//
//
//        // ===============================
//        // do sparse voting
//        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
//        sparse_ball_voting_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (ds_fieldarray, d_points, sigma, numPoints, d_log);
//        getLastCudaError("sparse_ball_voting_kernel() execution failed\n");
//        
//
//        // ===============================
//        // do tensor split
//        float3 * d_stick_field, * d_plate_field;
//        size_t sizeSparseEV = numPoints*sizeof(float3);
//        cutilSafeCall( cudaMalloc((void **)&d_stick_field, sizeSparseEV) );
//        cutilSafeCall( cudaMalloc((void **)&d_plate_field, sizeSparseEV) );
//        float * d_stick, * d_plate, * d_ball;
//        size_t sizeSaliency = numPoints*sizeof(float);
//        cutilSafeCall( cudaMalloc((void **)&d_stick, sizeSaliency) );
//        cutilSafeCall( cudaMalloc((void **)&d_plate, sizeSaliency) );
//        cutilSafeCall( cudaMalloc((void **)&d_ball, sizeSaliency) );
//
//        printf ("[GPU] deisgned size: %d\n", (numPoints-1)/BLOCK_DIM + 1);
//        tensor_split_with_eigenvectors_kernel <<< (numPoints-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (ds_fieldarray, d_stick, d_plate, d_ball,
//                    d_stick_field, d_plate_field, numPoints);
//        getLastCudaError("tensor_split_kernel() execution failed\n");
//
//
//        // ===============================
//        // clean un-needed sparse field
//        //cutilSafeCall( cudaFree(ds_fieldarray) );
//
//        // ===============================
//        // create grid for points
//        size_t sizePointGrid = numPoints*sizeof(float3);
//        float3 * d_pointgrid; // kept on GPU
//        cutilSafeCall( cudaMalloc((void **)&d_pointgrid, sizePointGrid) );
//        cutilSafeCall( cudaMemset(d_pointgrid, 0, sizePointGrid) );
//        // create grid for the points; get the number of cells needs to be created
//        // create d_pointgrid, saving only the nearest point
//        // d_pointgrid: [{int_x, int_y, int_z}]
//        printf ("[GPU] deisgned size for kernel d_pointgrid: %d\n", (numPoints-1)/BLOCK_DIM_G + 1);
//        real_pointgrid_creation <<< (numPoints-1)/BLOCK_DIM_G + 1, BLOCK_DIM_G >>> (d_pointgrid, d_points, 
//                    hw_size, cell_size, numPoints, 
//                    min_x, min_y, min_z, d_log);
//        getLastCudaError("real_dense_grid_creation() execution failed\n");
//
//        //=========================================
//        // do dense voting based on the grids created
//        size_t sizeGridField = maxGridSize*3*sizeof(float3); // change to output size
//        float3 * d_fieldarray;
//        cutilSafeCall( cudaMalloc((void **)&d_fieldarray, sizeGridField) );
//        cutilSafeCall( cudaMemset(d_fieldarray, 0, sizeGridField) );
//
//        printf ("[GPU] deisgned size for kernel real_dense_grid_voting(): %ld\n", (maxGridSize-1)/BLOCK_DIM + 1);
//        real_dense_plate_grid_voting <<< (maxGridSize-1)/BLOCK_DIM + 1, BLOCK_DIM >>> 
//            (d_fieldarray, d_pointgrid, d_stick_field, d_plate_field, d_plate, 
//             hw_size, cell_size, sigma, numPoints, maxGridSize, 
//             grid_dimx, grid_dimy, grid_dimz,
//             min_x, min_y, min_z, d_log);
//        getLastCudaError("real_dense_grid_voting() execution failed\n");
//        
//
//        // === release memory
//        cutilSafeCall( cudaFree(d_stick) );
//        cutilSafeCall( cudaFree(d_plate) );
//        cutilSafeCall( cudaFree(d_ball) );
//        cutilSafeCall( cudaFree(d_points) );
//        cutilSafeCall( cudaFree(d_log) );
//
//        cutilSafeCall( cudaFree(ds_fieldarray) );
//
//
//        //=========================================
//        // tensor split on d_fieldarray
//        // outputing saliency <- get number of cells in the grid first
//        float * dd_stick, * dd_plate, * dd_ball; //on-Device Dense (dd)
//        int numReal = maxGridSize; // TODO: find the real one
//        sizeSaliency = numReal*sizeof(float);
//        cutilSafeCall( cudaMalloc((void **)&dd_stick, sizeSaliency) );
//        cutilSafeCall( cudaMalloc((void **)&dd_plate, sizeSaliency) );
//        cutilSafeCall( cudaMalloc((void **)&dd_ball, sizeSaliency) );
//        printf ("[GPU] deisgned size for split dense field: %d\n", (numReal-1)/BLOCK_DIM + 1);
//        tensor_split_kernel<<< (numReal-1)/BLOCK_DIM + 1, BLOCK_DIM >>> (d_fieldarray, dd_stick, dd_plate, dd_ball,
//                    numReal);
//        getLastCudaError("tensor_split_kernel() execution failed\n");
//
//        cutilSafeCall( cudaMemcpy(stick_saliency_grid, dd_stick, sizeSaliency, cudaMemcpyDeviceToHost) );
//        cutilSafeCall( cudaMemcpy(plate_saliency_grid, dd_plate, sizeSaliency, cudaMemcpyDeviceToHost) );
//        cutilSafeCall( cudaMemcpy(ball_saliency_grid, dd_ball, sizeSaliency, cudaMemcpyDeviceToHost) );
//
//        //clean up
//        cutilSafeCall( cudaFree(dd_stick) );
//        cutilSafeCall( cudaFree(dd_plate) );
//        cutilSafeCall( cudaFree(dd_ball) );
//        cutilSafeCall( cudaFree(d_fieldarray) );
//
//#ifdef check_memory_size
//        cuCtxDetach( cudaContext );
//#endif
//        cutilSafeCall( cudaDeviceReset() );
//
//    }


} // end of namespace
