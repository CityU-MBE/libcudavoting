/*
 * =====================================================================================
 *
 *       Filename:  vtk_voting_example.cpp
 *
 *    Description:  test cuda tensor voting library with vtk input
 *
 *        Version:  1.0
 *        Created:  04/18/2012 07:55 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */

#include <iostream>

//#include "CudaVoting.h"
#include "assert.h"
//#include "vtktool.h"
// libpointmatcher
#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/Functions.h"
#include "aliases.h"

#include "tensor_voting.h"
#include "occupancygrid.h"
#include "vtktool.h"
#include "CudaVoting.h"

#include <time.h>
#include <cuda.h>
#include <cutil_math.h>
#include <vector_types.h>
#include <iostream>
#include <fstream>
#define USE_GPU_SPARSE_BALL

#define INT_ENDING 10000
//#define INT_ENDING 1000000
//
using namespace PointMatcherSupport;
using namespace std;
using namespace topomap;
using namespace cudavoting;

//! A dense matrix over ScalarType
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
//! A dense integer matrix
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> IntMatrix;

/* 
 * ===  FUNCTION  ============================================================
 *         Name:  Main  
 *  Description:  Entry of program
 * ===========================================================================
 */
int main ( int argc, char *argv[] )
{
    if (argc < 4)
    {
      cout << "Usage: ./fake_voting_dense_sparse_cpu <input.vtk> <output.vtk> sigma gridsize(1.0)" << endl;
      return 0;
    }
    cout << "Input file name: "<< argv[1];
    string filename(argv[1]);
    string outputname(argv[2]);

    DP input;
    input = PM::loadVTK(filename.c_str());
    cout << "Input size: " << input.features.size()/4 << endl;
    // subsampling here. Because the grid will otherwise too slow.
    PM pm;

    int numPoints = input.features.size()/4;

    // use CPU to vote for planes
    float sigma = 0.5;
    if(EOF == sscanf(argv[3], "%f", &sigma))
    {
        cout << "sigma not valid float." << endl;
        return 0;
    }
    cout << "Using sigma size: " << sigma << endl;



    TensorVoting voter(sigma); 
    cout << "Set point sets..."<< numPoints << endl;
    voter.setPoints(input);
#ifdef USE_GPU_SPARSE_BALL
    cout << "Sparse ball voting (GPU)..." << endl;
    Eigen::Matrix<Eigen::Matrix3f, Eigen::Dynamic, 1> sparseField;
//    CudaVoting::sparseBallVoting(sparseField, input.features, sigma);
//    float3 what = make_float3(1,2,3);

    // set output
    // ATTENTION: sm_11 only support atomicAdd on int, therefore here it need to be int
    // for sm_20 (hardware) or above, the code can be better adapted.
    // for compatiable reasons, here use int

    // float
//    size_t sizeField = numPoints*3*sizeof(float3);
//    float3* h_fieldarray = (float3 *)malloc(sizeField);
//    bzero( h_fieldarray, sizeField );
    // int
    size_t sizeField = numPoints*3*sizeof(int3);
    int3* h_fieldarray = (int3 *)malloc(sizeField);
//    bzero( h_fieldarray, sizeField );
//
//    for(int i = 0; i<numPoints*3; i++)
//    {
//        switch (i%3)
//        {
//            case 0:
//                h_fieldarray[i].x = 1*INT_ENDING;
//                break;
//            case 1:
//                h_fieldarray[i].y = 1*INT_ENDING;
//                break;
//            case 2:
//                h_fieldarray[i].z = 1*INT_ENDING;
//                break;
//        }
//    }
//    // check input field array
//    for(int i = 0; i<10*3; i++)
//    {
//        printf("%.2f %.2f %.2f\n", h_fieldarray[i].x, h_fieldarray[i].y, h_fieldarray[i].z);
//    }

    //set input:   getArrayFromMatrix();
    size_t sizePoints = numPoints*sizeof(float3);
    float3 *h_points = (float3 *)malloc(sizePoints);
    for(unsigned int i = 0; i<numPoints; i++)
    {
        h_points[i].x = input.features(0,i); 
        h_points[i].y = input.features(1,i); 
        h_points[i].z = input.features(2,i); 
    }

    // set log
    size_t sizeLog = numPoints*sizeof(int2);
    int2 * h_log = (int2 *)malloc(sizeLog);
    bzero( h_log, sizeLog);

    // call CUDA
    cout << "Send to GPU..." << endl;
    CudaVoting::sparseBallVoting(h_fieldarray, h_points, sigma, numPoints, h_log);
    // post-processing
    sparseField.resize(numPoints, 1);
    for(unsigned int i = 0; i<numPoints; i++)
    {
        Eigen::Matrix3f M;
        M << (float)(h_fieldarray[i*3 + 0].x)/INT_ENDING, (float)(h_fieldarray[i*3 + 0].y)/INT_ENDING, (float)(h_fieldarray[i*3 + 0].z)/INT_ENDING, 
             (float)(h_fieldarray[i*3 + 1].x)/INT_ENDING, (float)(h_fieldarray[i*3 + 1].y)/INT_ENDING, (float)(h_fieldarray[i*3 + 1].z)/INT_ENDING, 
             (float)(h_fieldarray[i*3 + 2].x)/INT_ENDING, (float)(h_fieldarray[i*3 + 2].y)/INT_ENDING, (float)(h_fieldarray[i*3 + 2].z)/INT_ENDING;
        sparseField(i) = M;
    }
    voter.setSparseTensor(sparseField);
    printf("Outputing..");

    ofstream out("log_fake_sparse_gpu1.txt");
    for(int i =0; i<numPoints; i++)
        //out << voter.sparseTensors(i) << endl;
        out << sparseField(i) << endl;
    out.close();


    // TODO: COMPARE THE NUMBER OF VOTEES PER VOTER

    ofstream out1("log_point_gpu1.txt");
    out1 << input.features << endl;
//    for(unsigned int i = 0; i<numPoints; i++)
//    {
//        out1 << h_points[i].x << '\t';
//    }
//    out1 << endl;
//    for(unsigned int i = 0; i<numPoints; i++)
//    {
//        out1 << h_points[i].y << '\t';
//    }
//    out1 << endl;
//    for(unsigned int i = 0; i<numPoints; i++)
//    {
//        out1 << h_points[i].z << '\t';
//    }
//    out1 << endl;
//    for(unsigned int i = 0; i<numPoints; i++)
//    {
//        out1 << 1 << '\t';
//    }
//    out1 << endl;
    out1.close();

    ofstream logf("log_number_of_whowhom_sparse_gpu1.txt");
    for(int i =0; i<numPoints; i++)
    {
        logf << i << " " << h_log[i].x << " " << h_log[i].y << endl;
    }
    logf.close();


    // clean up
    free(h_fieldarray);
    //free(h_points); //???????????????????????????
    free(h_log);

#else
    cout << "Sparse ball voting (CPU)..." << endl;
    voter.sparse_ball_vote();
    cout << endl;
#endif
    cout	<< "sparse tensor split..." << endl;
    voter.sparse_tensor_split();
    cout << endl;

    // save sparse tensor
    PM::DataPoints::Descriptors stick=PM::DataPoints::Descriptors::Zero(1, numPoints);
    PM::DataPoints::Descriptors plate=PM::DataPoints::Descriptors::Zero(1, numPoints);
    PM::DataPoints::Descriptors ball =PM::DataPoints::Descriptors::Zero(1, numPoints);
    for(unsigned int i=0; i<numPoints; i++)
    {
        stick(i) = voter.sparseStick(0,i);
        plate(i) = voter.sparsePlate(0,i*2);
        ball(i) =  voter.sparseBall(i);
    }

#if 1
    // write to pointcloud, then into vtk file
    input.addDescriptor("stick", stick);
    input.addDescriptor("plate", plate);
    input.addDescriptor("ball", ball);
    input.addDescriptor("normals", voter.sparseStick.bottomRows(3));
    PM::saveVTK(input, "sparse_gpu.vtk");
#endif



    /*-----------------------------------------------------------------------------
     *  Create Dense Grid
     *-----------------------------------------------------------------------------*/
    float length = 1.0;
    if (argc > 4)
      length = atof(argv[4]);
    cout << "Load vtk as grid..size: " << length << endl;
    Grid grid3d(filename, length, length, length, 1.0); // use original, 1.0=sample nothing
    VTKTool::dumpGridCell("original_freea.vtk", grid3d.freeCells);
    // get grid centers:
    size_t sizeGrid = grid3d.freeCells.size()*sizeof(float3);
    float3 *h_grids = (float3 *)malloc(sizeGrid);
    for(unsigned int i = 0; i<grid3d.freeCells.size(); i++)
    {
        float cx = grid3d.freeCells[i].pose.get<0>();
        float cy = grid3d.freeCells[i].pose.get<1>();
        float cz = grid3d.freeCells[i].pose.get<2>();
        h_grids[i] = make_float3(cx, cy, cz);
    }

    /*-----------------------------------------------------------------------------
     *  Dense Grid voting
     *-----------------------------------------------------------------------------*/

    cout << "Send to GPU dense stick voting..." << endl;
    // setZero of h_field
    size_t sizeDenseField = sizeGrid * 3;
    int3 * h_fieldarraydense = (int3 *)malloc(sizeDenseField);
    bzero( h_fieldarraydense, sizeDenseField);

    // copy to get h_sparsestick as input to GPU
    size_t sizeSparseStick = numPoints*4*sizeof(float4);
    float4 * h_sparsestick = (float4 *)malloc(sizeSparseStick);
    for(unsigned int i = 0; i<numPoints; i++)
    {
        h_sparsestick[i].x = voter.sparseStick(0,i);  // saliency
        h_sparsestick[i].y = voter.sparseStick(1,i);  // x
        h_sparsestick[i].z = voter.sparseStick(2,i);  // y
        h_sparsestick[i].w = voter.sparseStick(3,i);  // z
    }
    // set log
    int2 * hh_log = (int2 *)malloc(sizeLog);
    bzero( hh_log, sizeLog);

    // send to GPU
    cout << "Send to GPU dense stick voting... numPoints: "<< numPoints << " numGrids: " << grid3d.freeCells.size() << endl;
    CudaVoting::denseGridStickVoting(h_fieldarraydense, h_grids, h_points, h_sparsestick, sigma, grid3d.freeCells.size(), numPoints, hh_log);

    ofstream logff("log_number_of_whowhom_dense_gpu2.txt");
    for(int i =0; i<numPoints; i++)
    {
        logff << i << " " << hh_log[i].x << " " << hh_log[i].y << endl;
    }
    logff.close();

    // post-processing
    Eigen::Matrix<Eigen::Matrix3f, Eigen::Dynamic, 1> denseField;
    denseField.resize(grid3d.freeCells.size(), 1);
    for(unsigned int i = 0; i<grid3d.freeCells.size(); i++)
    {
        Eigen::Matrix3f M;
        M.setZero();
        // May need transpose (symmetric anyway)
        M << (float)(h_fieldarraydense[i*3 + 0].x)/INT_ENDING, (float)(h_fieldarraydense[i*3 + 0].y)/INT_ENDING, (float)(h_fieldarraydense[i*3 + 0].z)/INT_ENDING, 
             (float)(h_fieldarraydense[i*3 + 1].x)/INT_ENDING, (float)(h_fieldarraydense[i*3 + 1].y)/INT_ENDING, (float)(h_fieldarraydense[i*3 + 1].z)/INT_ENDING, 
             (float)(h_fieldarraydense[i*3 + 2].x)/INT_ENDING, (float)(h_fieldarraydense[i*3 + 2].y)/INT_ENDING, (float)(h_fieldarraydense[i*3 + 2].z)/INT_ENDING;
        denseField(i) = M;
    }
    printf("Outputing..");
    ofstream outdense("log_fake_dense_gpu2.txt");
    for(int i =0; i<grid3d.freeCells.size(); i++)
        outdense << denseField(i) << endl;
    outdense.close();

    // clean up
    free(h_fieldarraydense);
    free(h_sparsestick);
    free(h_points);
    free(h_grids);

    cout	<< "dense tensor split..." << endl;

    
    // explicit way of tensor decomposition:
    
    vector<TensorEigenValuesType> tensorVector;
    tensorVector.reserve(denseField.rows());
    /*-----------------------------------------------------------------------------
     *  Organize Tensor voting results
     *-----------------------------------------------------------------------------*/
    for(unsigned int i=0; i<denseField.rows(); i++)
    {
        printf("Extract eigen values: %d / %d\r", i, denseField.rows());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(denseField(i), false); // only calculate eigen values
        Eigen::Vector3f abs_evalues;
        Eigen::Vector3i eig_indices(0,0,0);

        abs_evalues = es.eigenvalues().array().abs();
        abs_evalues.maxCoeff(&eig_indices(0));
        abs_evalues.minCoeff(&eig_indices(2));
        for (unsigned int x=0; x<3; ++x)
        {
            if (x != eig_indices(0) && x!= eig_indices(2)) { eig_indices(1) = x; }
        }	
        TensorEigenValuesType tensorEigenvalues(
                    es.eigenvalues()[eig_indices(0)] - es.eigenvalues()[eig_indices(1)], // stick <- voter for planes
                    es.eigenvalues()[eig_indices(1)] - es.eigenvalues()[eig_indices(2)], // plate <- voter for curves
                    es.eigenvalues()[eig_indices(2)] // ball <- voter for single points
                    ); 

        tensorVector.push_back(tensorEigenvalues);
        //printf("where: %d \t Stick: %.3f \t Plate: %.3f \t Ball: %.3f\n", 
        //            i, tensorEigenvalues.get<0>(), tensorEigenvalues.get<1>(), tensorEigenvalues.get<2>());
    }
    stick=PM::DataPoints::Descriptors::Zero(1, denseField.rows());
    plate=PM::DataPoints::Descriptors::Zero(1, denseField.rows());
    ball =PM::DataPoints::Descriptors::Zero(1, denseField.rows());
    for(unsigned int i=0; i<tensorVector.size(); i++)
    {
        stick(i) = tensorVector[i].get<0>();
        plate(i) = tensorVector[i].get<1>();
        ball(i) = tensorVector[i].get<2>();
    }
    
    cout	<< "Write to file." << endl;
    VTKTool::dumpGridCell(outputname, grid3d.freeCells, stick, plate, ball);

//
    return 1;
}	/* ----------  end of function main  ---------- */
