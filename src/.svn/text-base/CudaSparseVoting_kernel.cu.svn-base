#ifndef CUDA_SPARSE_VOTING_KERNEL
#define CUDA_SPARSE_VOTING_KERNEL
namespace cudavoting{


    __global__ void sparse_ball_voting_kernel(float3 * field, const float3 * __restrict__ points, const float sigma, const int numPoints, int2 * logg)
    {
        __shared__ float3 tmpfield[BLOCK_DIM*3];
        int token = threadIdx.x+blockIdx.x*blockDim.x;

        tmpfield[threadIdx.x*3+0] = make_float3(1.0,0,0);
        tmpfield[threadIdx.x*3+1] = make_float3(0,1.0,0);
        tmpfield[threadIdx.x*3+2] = make_float3(0,0,1.0);
        __syncthreads();

        if (token >= numPoints) return;

        float3 votee = points[token];

        #pragma unroll 64
        for(unsigned int voter_i = 0; voter_i<numPoints; voter_i ++)
        {
            float3 voter = points[voter_i];
            if (token == voter_i) continue;

            float3 v = votee - voter;
            //float l = __powf(v.x,2) + __powf(v.y,2) + __powf(v.z,2);
            float l = pow(v.x,2) + pow(v.y,2) + pow(v.z,2);
            float z = __fdividef(__fsqrt_rn(l),sigma);
            if(l>0 && z<3)
            {
                // 40ms additional for 1000 points
                //logg[token].x += 1;
                //atomicAdd(&(logg[voter_i].y), 1); //logg[votee_i].y += 1;
                // outer product
                float3 vv[3];
                vv[0] = make_float3(pow(v.x,2), __fmul_rn(v.x,v.y), __fmul_rn(v.x,v.z));
                vv[1] = make_float3(__fmul_rn(v.y,v.x), pow(v.y,2), __fmul_rn(v.y,v.z));
                vv[2] = make_float3(__fmul_rn(v.z,v.x), __fmul_rn(v.z,v.y), pow(v.z,2));
                float norm_vv = __fsqrt_rn( 
                            pow(vv[0].x,2) + pow(vv[0].y,2) + pow(vv[0].z,2) +
                            pow(vv[1].x,2) + pow(vv[1].y,2) + pow(vv[1].z,2) +
                            pow(vv[2].x,2) + pow(vv[2].y,2) + pow(vv[2].z,2)
                            );
                // ATTENTION: sm_11 only support integer atomicAdd
                //float decay = FDIV(z*z*(z-3)*(z-3)*(z-3)*(z-3),16);
                float decay = __expf(-pow(z,2));

                float3 vv_use[3];
                vv_use[0] = make_float3( __fmul_rn(decay, 1 - __fdividef(vv[0].x,norm_vv)), __fmul_rn(decay, 0 - __fdividef(vv[0].y,norm_vv)), __fmul_rn(decay, 0 - __fdividef(vv[0].z,norm_vv)) );
                vv_use[1] = make_float3( __fmul_rn(decay, 0 - __fdividef(vv[1].x,norm_vv)), __fmul_rn(decay, 1 - __fdividef(vv[1].y,norm_vv)), __fmul_rn(decay, 0 - __fdividef(vv[1].z,norm_vv)) );
                vv_use[2] = make_float3( __fmul_rn(decay, 0 - __fdividef(vv[2].x,norm_vv)), __fmul_rn(decay, 0 - __fdividef(vv[2].y,norm_vv)), __fmul_rn(decay, 1 - __fdividef(vv[2].z,norm_vv)) );

                // update outputing field
                tmpfield[threadIdx.x*3 + 0] += vv_use[0];
                tmpfield[threadIdx.x*3 + 1] += vv_use[1];
                tmpfield[threadIdx.x*3 + 2] += vv_use[2];
            } // end of if
        }// end of for all voters
        __syncthreads();
        field[token*3 + 0] = tmpfield[threadIdx.x*3 + 0];
        field[token*3 + 1] = tmpfield[threadIdx.x*3 + 1];
        field[token*3 + 2] = tmpfield[threadIdx.x*3 + 2];
    }
}

#endif 
