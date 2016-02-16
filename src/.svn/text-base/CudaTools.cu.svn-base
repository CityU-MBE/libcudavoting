#ifndef CUDA_TOOLS_CU
#define CUDA_TOOLS_CU

namespace cudavoting{


    __device__ inline int3 toInt3(float3 A, int SCALE)
    {
        return make_int3((int)(A.x*SCALE), (int)(A.y*SCALE),(int)(A.z*SCALE));
    }
    __device__ inline void atomicAdd3( int * A, int3 B )
    {
         atomicAdd( A+0, B.x );
         atomicAdd( A+1, B.y );
         atomicAdd( A+2, B.z );
    }

    __device__ inline float3 product(float3 a, float b)
    {
        return make_float3( __fmul_rn(a.x, b), __fmul_rn(a.y,b), __fmul_rn(a.z, b));
    }

    __device__ inline float squaredNorm(float3 a)
    {
        return a.x*a.x + a.y*a.y + a.z*a.z;
    }
    __device__ inline float invSqrt(float x)// carmack algorithm
    {
        float xhalf = 0.5f*x;
        int i = *(int*)&x;
        i = 0x5f3759df - (i >> 1); // first order approx
        x = *(float*)&i;
        x = x*(1.5f - xhalf*x*x); // newton iteration
        x = x*(1.5f - xhalf*x*x); // newton iteration second time// can be removed
        return x;
    }







}
#endif
