// tensor split: parallel 3x3 matrix eigen decomposition

#ifndef CUDA_TENSOR_SPLIT_CU
#define CUDA_TENSOR_SPLIT_CU

#include "CudaMat33_kernel.cu"
namespace cudavoting{

    __global__ void tensor_split_kernel(const float3 * field, float * stick, float* plate, float * ball, int numPoints)
    {
        int token = threadIdx.x+blockIdx.x*blockDim.x;
        if (token >= numPoints) return;

        // target tensor matrix(33)
        float3 T[3];
        T[0] = field[token*3 + 0];
        T[1] = field[token*3 + 1];
        T[2] = field[token*3 + 2];
        
        // get eigenvalues
        float3 eigs;
        eigenvalues_33(eigs, T);

        // get abs values, and decreasing sort
        float3 eigs_abs = abs_3(eigs);
        
        // sort three numbers
        const float a=eigs_abs.x;
        const float b=eigs_abs.y;
        const float c=eigs_abs.z;
        float3 eigs_sorted = sort3_decrease(a,b,c);

        // write output;
        stick[token] = eigs_sorted.x - eigs_sorted.y;
        plate[token] = eigs_sorted.y - eigs_sorted.z;
        ball[token] = eigs_sorted.z;
    }

    // return saliency and normal direction
    __global__ void tensor_split_kernel_normal(const float3 * field, float * stick, float* plate, float * ball, float3 * normals, int numPoints)
    {
        int token = threadIdx.x+blockIdx.x*blockDim.x;
        if (token >= numPoints) return;

        // target tensor matrix(33)
        float3 T[3];
        T[0] = field[token*3 + 0];
        T[1] = field[token*3 + 1];
        T[2] = field[token*3 + 2];
        
        // get eigenvalues
        float3 abseigenvalues;
        float3 eigenvectors[3];
        eigen_33(abseigenvalues, eigenvectors, T); // !!! decreasingly sorted eigenvalues and corresponding eigenvectors (by abs. value)


        // get abs values, and decreasing sort
    //    float3 eigs_abs = abs_3(eigs);
        
        // sort three numbers
 //       const float a=eigs_abs.x;
 //       const float b=eigs_abs.y;
 //       const float c=eigs_abs.z;
 //       float3 eigs_sorted = sort3_decrease(a,b,c);

        // write output;
      //  stick[token] = eigs_sorted.x - eigs_sorted.y;
      //  plate[token] = eigs_sorted.y - eigs_sorted.z;
      //  ball[token] = eigs_sorted.z;

        // write saliency output
        stick[token] = abseigenvalues.x - abseigenvalues.y;
        plate[token] = abseigenvalues.y - abseigenvalues.z;
        ball[token] = abseigenvalues.z;

        // write field output
        normals[token] = eigenvectors[0];

    }

    __global__ void tensor_split_with_eigenvectors_kernel(const float3 * field, float * stick, float* plate, float * ball, 
                float3 * stick_field, float3 * plate_field, // note: ball_field is not necessary
                const int numPoints)
    {
        int token = threadIdx.x+blockIdx.x*blockDim.x;
        if (token >= numPoints) return;

        // target tensor matrix(33)
        float3 T[3];
        T[0] = field[token*3 + 0];
        T[1] = field[token*3 + 1];
        T[2] = field[token*3 + 2];
        
        // get eigenvalues
        float3 abseigenvalues;
        float3 eigenvectors[3];
        eigen_33(abseigenvalues, eigenvectors, T); // !!! decreasingly sorted eigenvalues and corresponding eigenvectors (by abs. value)

        // write saliency output
        stick[token] = abseigenvalues.x - abseigenvalues.y;
        plate[token] = abseigenvalues.y - abseigenvalues.z;
        ball[token] = abseigenvalues.z;

        // write field output
        stick_field[token] = eigenvectors[0];
        plate_field[token] = eigenvectors[1]; // secondary direction. Primary direction is the same as stick_field
    }
}

#endif
