// This file defines basic operations on 3x3 Matrix
// e.g. eigenvalue calculation by Smith's algorithm

#ifndef CUDA_MAT33_KERNEL_CU
#define CUDA_MAT33_KERNEL_CU
namespace cudavoting{

    __device__ float trace_33(const float3 mat[3])
    {
        return mat[0].x + mat[1].y + mat[2].z;
    }

    __device__ float det_33(const float3 K[3])
    {
        const float a=K[0].x;
        const float b=K[0].y;
        const float c=K[0].z;
        const float d=K[1].y;
        const float m=K[1].z;
        const float f=K[2].z;
        return a*d*f-a*m*m+b*b*(-f)+2*b*c*m-c*c*d;
    }

    __device__ float sum_33(const float3 K[3])
    {
        const float a=K[0].x;
        const float b=K[0].y;
        const float c=K[0].z;
        const float d=K[1].y;
        const float m=K[1].z;
        const float f=K[2].z;
        return a+b+c+b+d+m+c+m+f;
    }

    //C=A x B
    __device__ void product_33(const float3 A[3], const float3 B[3], float3 C[3])
    {
        const float a=A[0].x;
        const float b=A[0].y;
        const float c=A[0].z;
        const float d=A[1].y;
        const float m=A[1].z;
        const float f=A[2].z;

        const float h=B[0].x;
        const float g=B[0].y;
        const float p=B[0].z;
        const float n=B[1].y;
        const float t=B[1].z;
        const float q=B[2].z;

        C[0] = make_float3(b*g+a*h+c*p, a*g+b*n+c*t, a*p+c*q+b*t);
        C[1] = make_float3(d*g+b*h+m*p, b*g+d*n+m*t, b*p+m*q+d*t);
        C[2] = make_float3(c*h+g*m+f*p, c*g+m*n+f*t, c*p+f*q+m*t);
    }

    //C=A x B
    __device__ void product2_33(const float3 A[3], float3 C[3])
    {
        const float a=A[0].x;
        const float b=A[0].y;
        const float c=A[0].z;
        const float d=A[1].y;
        const float m=A[1].z;
        const float f=A[2].z;

        C[0]=make_float3(a*a+b*b+c*c,a*b+d*b+c*m,a*c+f*c+b*m);
        C[1]=make_float3(a*b+d*b+c*m,b*b+d*d+m*m,b*c+d*m+f*m);
        C[2]=make_float3(a*c+f*c+b*m,b*c+d*m+f*m,c*c+f*f+m*m);
    }

    //C=A.*A
    __device__ void dotproduct2_33(const float3 A[3], float3 C[3])
    {
        const float a=A[0].x;
        const float b=A[0].y;
        const float c=A[0].z;
        const float d=A[1].y;
        const float m=A[1].z;
        const float f=A[2].z;

        C[0]=make_float3(a*a, b*b, c*c);
        C[1]=make_float3(b*b, d*d, m*m);
        C[2]=make_float3(c*c, m*m, f*f);
    }

    __device__ float3 abs_3(const float3 A)
    {
        return make_float3(fabs(A.x), fabs(A.y), fabs(A.z));
    }

    __device__ float3 get_V_rank3(float3 M[3])
    {
        // find two independent rows of M0
        float rx = M[0].x / M[1].x;
        float ry = M[0].y / M[1].y;
        float rz = M[0].z / M[1].z;

        float3 r0, r1;
        if(fabs(rx - ry) < 1e4 && fabs(ry - rz)<1e4) // same ratio
        {// [0] and [1] are dependent, just take [0] and [2]
            r0 = M[0]; r1 = M[2];
        }
        else
        {
            r0 = M[0]; r1 = M[1];
        }
        return normalize( cross(r0, r1) );
    }


    __device__ void get_V_rank2(float3 M[3], float3& V, float3& W) // multiplicity 2
    {
        float3 U = normalize(M[0]);
        if ( fabs(U.x) >= fabs(U.y) )
        {
            float invLength = 1.0/sqrt(U.x*U.x + U.z*U.z);
            V.x = -U.z*invLength;
            V.y = 0;
            V.z = U.x*invLength;
            W.x = U.y*V.z;
            W.y = U.z*V.x - U.x*V.z;
            W.z = -U.y*V.x;
        }
        else
        {
            float invLength = 1.0/sqrt(U.y*U.y + U.z*U.z);
            V.x = 0;
            V.y = U.z*invLength;
            V.z = -U.y*invLength;
            W.x = U.y*V.z - U.z*V.y;
            W.y = -U.x*V.z;
            W.z = U.x*V.y;
        }
    }
    __device__ float3 sort3_decrease(const float a, const float b, const float c)
    {
        if (a>b) 
        {
            if (a>c)
            {
                if(b>c)
                {
                    return make_float3(a,b,c);
                }
                else
                {
                    return make_float3(a,c,b);
                }
            }
            else
            {
                return make_float3(c,a,b);
            }
        } 
        else  // a <= b
        {
            if (b>c)
            {
                if (a>c)
                {
                    return make_float3(b,a,c);
                }
                else
                {
                    return make_float3(b,c,a);
                }
            }
            else
            {
                return make_float3(c,b,a);
            }
        }
    
    }

    // eigenvalues for 3x3 matrix
    __device__ void eigenvalues_33(float3 & eig, const float3 M[3])
    {
        float m = trace_33(M)*(1.0/3);
        float3 K[3];
        K[0] = M[0] - make_float3(m, 0, 0);
        K[1] = M[1] - make_float3(0, m, 0);
        K[2] = M[2] - make_float3(0, 0, m);
        float q = det_33(K)*0.5;
        float3 prd[3];
        dotproduct2_33(K,prd);
        float p = sum_33(prd) * (1.0/6);
        float phi = (1.0/3.0)*acos( q* (1.0/ powf(p,(3.0/2.0))) );

        if( fabs(q) >= fabs(powf(p,(3.0/2.0))) )
          phi = 0;

        if( phi < 0 )
          phi = phi + 3.14159265/3.0;

        eig.x = m + 2*__fsqrt_rn(p)*__cosf(phi);
        eig.y = m - __fsqrt_rn(p)*( __cosf(phi) + __fsqrt_rn(3)*__sinf(phi) );
        eig.z = m - __fsqrt_rn(p)*( __cosf(phi) - __fsqrt_rn(3)*__sinf(phi) );

      //  printf ( "eigs: [ %f \t %f \t %f]\n", eig.x, eig.y, eig.z );
    }

    // eigenvectors for 3x3 matrix; 
    // outputing sorted eigenvalues: float3 eig, by decreasing absolute values
    // outputing eigenvectors corresponding to eig
    __device__ void eigen_33(float3 & abseigenvalues, float3 eigenvectors[3], const float3 M[3] )
    {
        // get eigenvalue
        float3 eigs;
        eigenvalues_33(eigs, M);
        const float l0 = eigs.z; //small
        const float l1 = eigs.y;
        const float l2 = eigs.x;

        // get eigenvectors
        float3 M0[3]; // row vectors
        M0[0] = M[0] - make_float3(l0, 0, 0);
        M0[1] = M[1] - make_float3(0, l0, 0);
        M0[2] = M[2] - make_float3(0, 0, l0);
        float3 M1[3];
        M1[0] = M[0] - make_float3(l1, 0, 0);
        M1[1] = M[1] - make_float3(0, l1, 0);
        M1[2] = M[2] - make_float3(0, 0, l1);
        float3 M2[3];
        M2[0] = M[0] - make_float3(l2, 0, 0);
        M2[1] = M[1] - make_float3(0, l2, 0);
        M2[2] = M[2] - make_float3(0, 0, l2);

        // http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf

        float3 ev[3]; // eigenvector
        // start eigenvector calculation
        if (l0 == l1 && l1 == l2) // rank1
        {
            ev[0]=make_float3(1,0,0);
            ev[1]=make_float3(0,1,0);
            ev[2]=make_float3(0,0,1);
        }
        // rank2
        else if (l0==l1 && l1<l2)
        {
            ev[2] = get_V_rank3(M2); // multiplicity 1, can use get_V_rank3 directly
            get_V_rank2(M0, ev[1], ev[0]);
        }
        else // rank3, all not equal : l0 <l1 < l2
        {
            ev[0] = get_V_rank3(M0); // correspond to smallest eigenvalue
            ev[1] = get_V_rank3(M1);
            ev[2] = get_V_rank3(M2);
        }


        // sort three absolute numbers
        float3 eigs_abs = abs_3(eigs);
        const float a=eigs_abs.x;
        const float b=eigs_abs.y;
        const float c=eigs_abs.z;
        abseigenvalues = sort3_decrease(a,b,c);

        // find corresponding relations
        if (fabs(l0) == abseigenvalues.x)
            eigenvectors[0] = ev[0];
        else if (fabs(l0) == abseigenvalues.y)
            eigenvectors[1] = ev[0];
        else if (fabs(l0) == abseigenvalues.z)
            eigenvectors[2] = ev[0];
        else
            ; // not possible. must be at least one

        if (fabs(l1) == abseigenvalues.x)
            eigenvectors[0] = ev[1];
        else if (fabs(l1) == abseigenvalues.y)
            eigenvectors[1] = ev[1];
        else if (fabs(l1) == abseigenvalues.z)
            eigenvectors[2] = ev[1];
        else
            ; // not possible. must be at least one

        if (fabs(l2) == abseigenvalues.x)
            eigenvectors[0] = ev[2];
        else if (fabs(l2) == abseigenvalues.y)
            eigenvectors[1] = ev[2];
        else if (fabs(l2) == abseigenvalues.z)
            eigenvectors[2] = ev[2];
        else
            ; // not possible. must be at least one
    } //end of function

}
#endif
