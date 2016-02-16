#ifndef CUDA_DENSE_VOTING_KERNEL
#define CUDA_DENSE_VOTING_KERNEL

namespace cudavoting{


    __global__ void dense_grid_stick_voting_kernel(float3 * field, const
                float4* sparsestick, const float3 * grids, 
                const float3 * points, const float sigma, const unsigned int
                numGrids, const int numPoints, int2 * logg)
    {
        __shared__ float3 tmpfield[BLOCK_DIM*3];
        int token = threadIdx.x+blockIdx.x*blockDim.x;
        tmpfield[threadIdx.x*3+0] = make_float3(0,0,0);
        tmpfield[threadIdx.x*3+1] = make_float3(0,0,0);
        tmpfield[threadIdx.x*3+2] = make_float3(0,0,0);
        __syncthreads();
        if (token >= numGrids) return;

        //float3 P = points[token]; // votee
        float3 P = grids[token]; // votee
        #pragma unroll 64
        for(unsigned int voter_i = 0; voter_i<numPoints; voter_i ++)
        {
            float4 _sparseStick = sparsestick[voter_i];
            float stick_saliency = _sparseStick.x;
            float3 vn = make_float3(_sparseStick.y, _sparseStick.z, _sparseStick.w);
           
#ifdef fast_invsqrt
            float _squaredn = squaredNorm(vn);
            if(_squaredn > 0)
                vn = product(vn, invSqrt(_squaredn));
#else
            float _norm = length(vn);
            if (_norm > 0) 
                vn = make_float3( __fdividef(vn.x, _norm), __fdividef(vn.y, _norm), __fdividef(vn.z, _norm));
#endif

            float3 vt;
            float3 vc;

            if (token == voter_i) continue;

            float3 O = points[voter_i]; // voter
            float3 v = P - O;// votee - voter;
            float l = length(v);
            float scaled_dist = __fdividef(l,sigma);//sqrt(l)/sigma;
            float DF = 0;
            if(l>0 && scaled_dist<3)
            {
                v = make_float3( __fdividef(v.x, l), __fdividef(v.y, l), __fdividef(v.z, l));
                float vvn = __fmul_rn(v.x, vn.x) + __fmul_rn(v.y, vn.y) + __fmul_rn(v.z, vn.z); //dot(v, vn);
                if (vvn < 0) vn = -vn;
                float theta = asin(vvn);

                if (fabs(theta) <= M_PI/4 || fabs(theta) >=3*M_PI/4)
                {
                    
                        //write_kernel_log: PASS :)
                        //logg[token].x += 1;
                        //atomicAdd(&(logg[voter_i].y), 1);
                    float3 crossvvn = make_float3(__fmul_rn(v.y, vn.z) - __fmul_rn(v.z, vn.y),
                                                  __fmul_rn(v.z, vn.x) - __fmul_rn(v.x, vn.z),
                                                  __fmul_rn(v.x, vn.y) - __fmul_rn(v.y, vn.x)
                                                  );
                    vt= make_float3(__fmul_rn(vn.y, crossvvn.z) - __fmul_rn(vn.z, crossvvn.y),
                                    __fmul_rn(vn.z, crossvvn.x) - __fmul_rn(vn.x, crossvvn.z),
                                    __fmul_rn(vn.x, crossvvn.y) - __fmul_rn(vn.y, crossvvn.x)
                                    );

#ifdef fast_invsqrt
                    float _vtl2= squaredNorm(vt);
                    if(_vtl2> 0)
                      vt = product(vt, invSqrt(_vtl2));
#else
                    float vtl = length(vt);
                    if (vtl > 0)
                        vt = make_float3( __fdividef(vt.x, vtl), __fdividef(vt.y, vtl), __fdividef(vt.z, vtl));
#endif
                    float vvt = __fmul_rn(v.x, vt.x) + __fmul_rn(v.y, vt.y) + __fmul_rn(v.z, vt.z);
                    if (vvt < 0) vt = -vt;

                    vc = product(vn, __cosf(2*theta)) - product(vt, __sinf(2*theta));

                    float c = pow(scaled_dist, 2)*pow(scaled_dist-3, 4)*(1.0/16.0); // smooth version

                    // WARNING: ERROR: __powf is very very LIMITED!!!!!!!!!

                    float r = length(vc);
                    //DF=exp( -(4*r*r*theta*theta + c/(r*r)) * (1.0/(sigma*sigma)) );
                    DF=exp( -(4*r*r*theta*theta + c/(r*r)) * (1.0/(sigma*sigma)) ) * pow(cos(theta), 8);

                    if (token == 8)
                        logg[voter_i].x = theta*10000; // success
                    if (token == 200)
                        logg[voter_i].y = theta*10000; // failed: There are some angles not 0, which causes low DF


                    // components
                    //Sp = DF*(vc.transpose()*vc);
                    float3 Sp[3];

                    Sp[0] = product(product(make_float3(vc.x*vc.x, vc.x*vc.y, vc.x*vc.z), DF), stick_saliency);
                    Sp[1] = product(product(make_float3(vc.y*vc.x, vc.y*vc.y, vc.y*vc.z), DF), stick_saliency);
                    Sp[2] = product(product(make_float3(vc.z*vc.x, vc.z*vc.y, vc.z*vc.z), DF), stick_saliency);

                    tmpfield[threadIdx.x*3 + 0] = Sp[0] + tmpfield[threadIdx.x*3 + 0] ;
                    tmpfield[threadIdx.x*3 + 1] = Sp[1] + tmpfield[threadIdx.x*3 + 1];
                    tmpfield[threadIdx.x*3 + 2] = Sp[2] + tmpfield[threadIdx.x*3 + 2];
                } // end of if angular
            } //end of if scale
        
        }// end of for all voters
        __syncthreads();
        field[token*3 + 0] = tmpfield[threadIdx.x*3 + 0];
        field[token*3 + 1] = tmpfield[threadIdx.x*3 + 1];
        field[token*3 + 2] = tmpfield[threadIdx.x*3 + 2];
    
    }



    // Voting on points instead of grids
    // For concept tests. DEPRECATED
    __global__ void dense_stick_voting_kernel(float3 * field, const float4 *
                sparsestick, const float3 * points, const float sigma, const
                int numPoints, int2 *logg)
    {
        __shared__ float3 tmpfield[BLOCK_DIM*3];
        int token = threadIdx.x+blockIdx.x*blockDim.x;
        tmpfield[threadIdx.x*3+0] = make_float3(0,0,0);
        tmpfield[threadIdx.x*3+1] = make_float3(0,0,0);
        tmpfield[threadIdx.x*3+2] = make_float3(0,0,0);
        __syncthreads();
        if (token >= numPoints) return;

        float3 P = points[token]; // votee
        #pragma unroll 64
        for(unsigned int voter_i = 0; voter_i<numPoints; voter_i ++)
        {
            float4 _sparseStick = sparsestick[voter_i];
            float stick_saliency = _sparseStick.x;
            float3 vn = make_float3(_sparseStick.y, _sparseStick.z, _sparseStick.w);
           
#ifdef fast_invsqrt
            float _squaredn = squaredNorm(vn);
            if(_squaredn > 0)
                vn = product(vn, invSqrt(_squaredn));
#else
            float _norm = length(vn);
            if (_norm > 0) 
                vn = make_float3( __fdividef(vn.x, _norm), __fdividef(vn.y, _norm), __fdividef(vn.z, _norm));
#endif

            float3 vt;
            float3 vc;

            if (token == voter_i) continue;

            float3 O = points[voter_i]; // voter
            float3 v = P - O;// votee - voter;
            float l = length(v);
            float scaled_dist = __fdividef(l,sigma);//sqrt(l)/sigma;
            float DF = 0;
            if(l>0 && scaled_dist<3)
            {
                v = make_float3( __fdividef(v.x, l), __fdividef(v.y, l), __fdividef(v.z, l));
                float vvn = __fmul_rn(v.x, vn.x) + __fmul_rn(v.y, vn.y) + __fmul_rn(v.z, vn.z); //dot(v, vn);
                if (vvn < 0) vn = -vn;
                float theta = asin(vvn);

                if (fabs(theta) <= M_PI/4 || fabs(theta) >=3*M_PI/4)
                {
                    
                    float3 crossvvn = make_float3(__fmul_rn(v.y, vn.z) - __fmul_rn(v.z, vn.y),
                                                  __fmul_rn(v.z, vn.x) - __fmul_rn(v.x, vn.z),
                                                  __fmul_rn(v.x, vn.y) - __fmul_rn(v.y, vn.x)
                                                  );
                    vt= make_float3(__fmul_rn(vn.y, crossvvn.z) - __fmul_rn(vn.z, crossvvn.y),
                                    __fmul_rn(vn.z, crossvvn.x) - __fmul_rn(vn.x, crossvvn.z),
                                    __fmul_rn(vn.x, crossvvn.y) - __fmul_rn(vn.y, crossvvn.x)
                                    );

#ifdef fast_invsqrt
                    float _vtl2= squaredNorm(vt);
                    if(_vtl2> 0)
                      vt = product(vt, invSqrt(_vtl2));
#else
                    float vtl = length(vt);
                    if (vtl > 0)
                        vt = make_float3( __fdividef(vt.x, vtl), __fdividef(vt.y, vtl), __fdividef(vt.z, vtl));
#endif
                    float vvt = __fmul_rn(v.x, vt.x) + __fmul_rn(v.y, vt.y) + __fmul_rn(v.z, vt.z);
                    if (vvt < 0) vt = -vt;

                    //vc = vn*cos(2*theta) - vt*sin(2*theta);
                    //vc = vn*__cosf(2*theta) - vt*__sinf(2*theta); //TODO: may also be optimized; Main source of error?
                    vc = product(vn, __cosf(2*theta)) - product(vt, __sinf(2*theta));




                    //float c = FDIV(scaled_dist*scaled_dist*(scaled_dist-3)*(scaled_dist-3)*(scaled_dist-3)*(scaled_dist-3),16);
                    //float c = __expf(-__powf(scaled_dist,2));
                    float c = pow(scaled_dist, 2)*pow(scaled_dist-3, 4)*(1.0/16.0); // smooth version
//                    float c = __fdividef(pow(scaled_dist, 2)*pow(scaled_dist-3, 4), 16); // smooth version

                    // WARNING: ERROR: __powf is very very LIMITED!!!!!!!!!
                    //float c = __fdividef(__powf(scaled_dist,2)*__powf(scaled_dist-3, 4), 16); 

                    //float r = __fsqrt_rn(__powf(vc.x,2) + __powf(vc.y,2) + __powf(vc.z,2));//length(vc);
                    float r = length(vc);
                    DF=exp( -(4*r*r*theta*theta + c/(r*r)) * (1.0/(sigma*sigma)) ) * pow(cos(theta), 8);
                    //DF = __expf( - __fdividef(4*__powf(r,2)*__powf(theta,2) + __fdividef(c, __powf(r,2)), __powf(sigma,2) ) ); 

                    // components
                    //Sp = DF*(vc.transpose()*vc);
                    float3 Sp[3];
                    //Sp[0] = stick_saliency*DF*make_float3(vc.x*vc.x, vc.x*vc.y, vc.x*vc.z);
                    //Sp[1] = stick_saliency*DF*make_float3(vc.y*vc.x, vc.y*vc.y, vc.y*vc.z);
                    //Sp[2] = stick_saliency*DF*make_float3(vc.z*vc.x, vc.z*vc.y, vc.z*vc.z);
                    Sp[0] = product(product(make_float3(vc.x*vc.x, vc.x*vc.y, vc.x*vc.z), DF), stick_saliency);
                    Sp[1] = product(product(make_float3(vc.y*vc.x, vc.y*vc.y, vc.y*vc.z), DF), stick_saliency);
                    Sp[2] = product(product(make_float3(vc.z*vc.x, vc.z*vc.y, vc.z*vc.z), DF), stick_saliency);

                    tmpfield[threadIdx.x*3 + 0] = Sp[0] + tmpfield[threadIdx.x*3 + 0] ;
                    tmpfield[threadIdx.x*3 + 1] = Sp[1] + tmpfield[threadIdx.x*3 + 1];
                    tmpfield[threadIdx.x*3 + 2] = Sp[2] + tmpfield[threadIdx.x*3 + 2];
                } // end of if angular
            } //end of if scale
        
        }// end of for all voters
        __syncthreads();
        field[token*3 + 0] = tmpfield[threadIdx.x*3 + 0];
        field[token*3 + 1] = tmpfield[threadIdx.x*3 + 1];
        field[token*3 + 2] = tmpfield[threadIdx.x*3 + 2];
    }



 __global__ void real_dense_stick_grid_voting(float3 * field, const float3 * d_pointgrid, 
                const float3 * sparsestick, const float * sticks, //new
                const int hw_size, const float cell_size,
                const float sigma, const int numPoints, 
                const unsigned int maxGridSize, 
                const unsigned int grid_dimx,
                const unsigned int grid_dimy,
                const unsigned int grid_dimz,
                const float min_x,
                const float min_y,
                const float min_z,
                int2 * logg)
    {

        __shared__ float3 tmpfield[BLOCK_DIM*3];
        tmpfield[threadIdx.x*3+0] = make_float3(0,0,0);
        tmpfield[threadIdx.x*3+1] = make_float3(0,0,0);
        tmpfield[threadIdx.x*3+2] = make_float3(0,0,0);
        __syncthreads();

        int token = threadIdx.x+blockIdx.x*blockDim.x; // over all points
        if (token >= maxGridSize) return;

        // votee P : position cell coordinate
        int Pz = token / (grid_dimx*grid_dimy);
        int Pxy = token % (grid_dimx*grid_dimy);
        int Px = Pxy % grid_dimx;
        int Py = Pxy / grid_dimx; // coord origined locally

        // global coord
        float3 P = make_float3((float)Px, (float)Py, (float)Pz );

        #pragma unroll 64
        for(unsigned int voter_i = 0; voter_i < numPoints; voter_i++)
        {
            float3 O_int = d_pointgrid[voter_i];

            // test O_int : PASS
 //           logg[voter_i].x = O_int.x;
 //           logg[voter_i].y = O_int.y;

            float3 O = make_float3((float)O_int.x, (float)O_int.y, (float)O_int.z);

            if ( abs(Px - O_int.x) > hw_size || abs(Py - O_int.y) > hw_size || abs(Pz - O_int.z) > hw_size)
              continue;
            
            float3 v = (P - O)*cell_size;
            float l = length(v);
            float scaled_dist = __fdividef(l,sigma);//sqrt(l)/sigma;
            if(l == 0 || scaled_dist >=3)
              continue;

            // old
            //float4 _sparseStick = sparsestick[voter_i];
            //float stick_saliency = _sparseStick.x;
            //float3 vn = make_float3(_sparseStick.y, _sparseStick.z, _sparseStick.w);

            // new
            float3 vn = sparsestick[voter_i];
            float stick_saliency = sticks[voter_i];

            float _norm = length(vn);
            if (_norm > 0) 
                vn = make_float3( __fdividef(vn.x, _norm), __fdividef(vn.y, _norm), __fdividef(vn.z, _norm));

            v = make_float3( __fdividef(v.x, l), __fdividef(v.y, l), __fdividef(v.z, l));
            float vvn = __fmul_rn(v.x, vn.x) + __fmul_rn(v.y, vn.y) + __fmul_rn(v.z, vn.z); //dot(v, vn);
            if (vvn < 0) vn = -vn;
            float theta = asin(vvn);

            if (fabs(theta) <= M_PI/4 || fabs(theta) >=3*M_PI/4)
            {
                
                float3 crossvvn = make_float3(__fmul_rn(v.y, vn.z) - __fmul_rn(v.z, vn.y),
                                              __fmul_rn(v.z, vn.x) - __fmul_rn(v.x, vn.z),
                                              __fmul_rn(v.x, vn.y) - __fmul_rn(v.y, vn.x)
                                              );
                float3 vt= make_float3(__fmul_rn(vn.y, crossvvn.z) - __fmul_rn(vn.z, crossvvn.y),
                                __fmul_rn(vn.z, crossvvn.x) - __fmul_rn(vn.x, crossvvn.z),
                                __fmul_rn(vn.x, crossvvn.y) - __fmul_rn(vn.y, crossvvn.x)
                                );
                float vtl = length(vt);
                if (vtl > 0)
                    vt = make_float3( __fdividef(vt.x, vtl), __fdividef(vt.y, vtl), __fdividef(vt.z, vtl));
                float vvt = __fmul_rn(v.x, vt.x) + __fmul_rn(v.y, vt.y) + __fmul_rn(v.z, vt.z);
                if (vvt < 0) vt = -vt;

                float3 vc = product(vn, __cosf(2*theta)) - product(vt, __sinf(2*theta));
                float c = pow(scaled_dist, 2)*pow(scaled_dist-3, 4)*(1.0/16.0); // smooth version
                float r = length(vc);
                float DF=exp( -(4*r*r*theta*theta + c/(r*r)) * (1.0/(sigma*sigma)) ) * pow(__cosf(theta), 8);
                float3 Sp[3];
                Sp[0] = product(product(make_float3(vc.x*vc.x, vc.x*vc.y, vc.x*vc.z), DF), stick_saliency);
                Sp[1] = product(product(make_float3(vc.y*vc.x, vc.y*vc.y, vc.y*vc.z), DF), stick_saliency);
                Sp[2] = product(product(make_float3(vc.z*vc.x, vc.z*vc.y, vc.z*vc.z), DF), stick_saliency);
                tmpfield[threadIdx.x*3 + 0] = Sp[0] + tmpfield[threadIdx.x*3 + 0] ;
                tmpfield[threadIdx.x*3 + 1] = Sp[1] + tmpfield[threadIdx.x*3 + 1];
                tmpfield[threadIdx.x*3 + 2] = Sp[2] + tmpfield[threadIdx.x*3 + 2];
            } // end of angle condition
        }// end of for all voters
        __syncthreads();
        field[token*3 + 0] = tmpfield[threadIdx.x*3 + 0];
        field[token*3 + 1] = tmpfield[threadIdx.x*3 + 1];
        field[token*3 + 2] = tmpfield[threadIdx.x*3 + 2];
    }










}
#endif
