/*
 * =====================================================================================
 *
 *       Filename:  tensor_voting.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03/30/2012 04:27:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */

#ifndef TENSOR_VOTING_H
#define TENSOR_VOTING_H
#include "global.h"
#include "aliases.h"
#include "boost/tuple/tuple.hpp"
#include <boost/tuple/tuple_comparison.hpp> // for map comparison
#include <Eigen/Eigen>
#include <vector>
#include <map>

#include <vector_types.h>

using namespace std;
using namespace Eigen;
namespace topomap{

    class TensorVoting
    {
        public:
            TensorVoting(float _sigma):sigma(_sigma)
            {
              //  c = -16*log(0.1)*(_sigma-1)/(M_PI*M_PI);
            }
            ~TensorVoting(){
            }

            // parameters:
            float sigma; // kernel size 
            float c; // decay parameter

            // type defines
            typedef boost::tuple< float, float, float > NormalType;
            typedef vector<NormalType> NormalsType;
            typedef boost::tuple< int, int, int > PoseIDType;
            typedef vector<PoseIDType> LinkedObstacleType;
            typedef vector<LinkedObstacleType> LinkedObstaclesType; // not good naming

            inline float radial_decay_smooth(const float z)
            {
                if (z<3)
                  return pow(z,2)*pow(z-3,4)/16;
                else
                  return 0;
            }

            inline float radial_decay_traditional(const float z)
            {
                return exp(-pow(z,2));
            }

            inline float angular_decay(const float theta)
            {
                return pow(cos(theta),8);
            }

            // variables:
            MatrixXf points;
            MatrixXf normals;
            Matrix<Matrix3f, Dynamic, 1> field;
            Matrix<Matrix3f,Dynamic,1> sparseTensors;
            Matrix<float,4,Dynamic> sparseStick;
            Matrix<float,4,Dynamic> sparsePlate;
            VectorXf sparseBall;

            // handlers:
            void setPoints( const LinkedObstacleType & pointsVector);
            void setPoints( const DP & pointsVector);
            void setSparseTensor(const Matrix<Matrix3f,Dynamic,1> _sparseTensor);
            void setNormals( const NormalsType & normalsVector);
            Matrix<Matrix3f, Dynamic, 1> stickDenseVote();
            Matrix<Matrix3f, Dynamic, 1> plateDenseVote();

            Matrix<Matrix3f,Dynamic,1> sparse_ball_vote ();
            void sparse_tensor_split ();
            int2 * h_log;
    };


}


#endif
