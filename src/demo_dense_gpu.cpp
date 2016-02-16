/*
 * =====================================================================================
 *
 *       Filename:  vtk_voting_example.cpp
 *
 *    Description:  load pointcloud2 and do sparsevoting
 *                  write back pointcloud2 fields
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

// system level
#include <iostream>
#include <fstream>
#include <memory>
#include <time.h>
#include <assert.h>

// CUDA related
#include <cuda.h>
#include <helper_math.h>
#include <vector_types.h>

// ROS related
#include "ros/ros.h"
#include "ros/console.h"
#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/IO.h"
#include "pointmatcher_ros/point_cloud.h"
#include "pointmatcher_ros/transform.h"
#include "get_params_from_server.h"

// libpointmatcher
#include "aliases.h"

// libCudaVoting
#include "tensor_voting.h"
#include "CudaVoting.h"

#define USE_GPU_SPARSE_BALL

using namespace PointMatcherSupport;
using namespace std;
using namespace topomap;
using namespace cudavoting;

//! A dense matrix over ScalarType
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
//! A dense integer matrix
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> IntMatrix;

class DenseVotingCloudGPU
{
    ros::NodeHandle & n;
    // subscriber
	ros::Subscriber cloudSub;
	string cloudTopicIn;
    // publisher
	ros::Publisher cloudPub;
	string cloudTopicOut;

	DP cloud; // input point-cloud
    DP grid;  // output grid

    // tensor voting
    float sigma; // sparse voting kernel size
    float cell_size; // grid cell size
    float hw; // extension of the calculated area, in terms of hw*sigma (e.g. hw=1.0~3.0)

    // parameter
    bool savevtk; // whether save sequence of vtk files.
	const string mapFrame;
    float TH_VALID_BALL, TH_VALID_STICK, TH_VALID_PLATE; // for visualization. If the saliency is smaller
                                            // than these given TH, the cell will be removed from result.

public:
	DenseVotingCloudGPU(ros::NodeHandle& n);
	void gotCloud(const sensor_msgs::PointCloud2& cloudMsg);
    void publish(const DP & cloud); // publish tensor voting result
    void process(DP & cloud, float sigma, float cell_size); // return the grid containing dense voting result
};

DenseVotingCloudGPU::DenseVotingCloudGPU(ros::NodeHandle& n):
    n(n),
	mapFrame(getParam<string>("mapFrameId", "/map"))
{
	// ROS initialization
    sigma = getParam<double>("sigma", 1.0);
    hw = getParam<double>("halfwindow", 1.0);
    cell_size = getParam<double>("cell_size", 1.0);
    TH_VALID_STICK = getParam<double>("TH_VALID_STICK", 1e-2);
    TH_VALID_PLATE = getParam<double>("TH_VALID_PLATE", 1e-2);
    TH_VALID_BALL = getParam<double>("TH_VALID_BALL", 1e-2);
	cloudTopicIn = getParam<string>("cloudTopicIn", "/point_cloud");
	cloudTopicOut = getParam<string>("cloudTopicOut", "/point_cloud_out");
    savevtk = getParam<bool>("savevtk", false);

	cloudSub = n.subscribe(cloudTopicIn, 100, &DenseVotingCloudGPU::gotCloud, this);
	cloudPub = n.advertise<sensor_msgs::PointCloud2>(
		getParam<string>("cloudTopicOut", "/point_cloud_densevoting"), 1
	);
}


void DenseVotingCloudGPU::gotCloud(const sensor_msgs::PointCloud2& cloudMsgIn)
{
    cloud = DP(PointMatcher_ros::rosMsgToPointMatcherCloud<float>(cloudMsgIn));

    // do sparse then dense tensor voting
    process(cloud, sigma, cell_size);

    ROS_INFO("output Cloud descriptor pointcloud size: %d", 
                (unsigned int)(cloud.features.cols()));

    // publishing
    publish(grid);

    if(savevtk)
    {
	    stringstream nameStream;
	    nameStream << "." << cloudTopicIn << "_" << cloudMsgIn.header.seq;
	    PointMatcherIO<float>::saveVTK(grid, nameStream.str());
    }
}

void DenseVotingCloudGPU::publish(const DP& cloud)
{
    cout << cloud.features << endl;
	if (cloudPub.getNumSubscribers())
		cloudPub.publish(PointMatcher_ros::pointMatcherCloudToRosMsg<float>(cloud, mapFrame, ros::Time::now()));
}

void DenseVotingCloudGPU::process(DP & cloud, float sigma, float cell_size)
{
    unsigned int numPoints = cloud.features.size()/4;
    cout << "Input size: " << numPoints << endl;
    cout << "Sparse ball voting (GPU)..." << endl;

    // 1. allocate field
    Eigen::Matrix<Eigen::Matrix3f, Eigen::Dynamic, 1> sparseField;
    size_t sizeField = numPoints*3*sizeof(float3);
    float3* h_fieldarray = (float3 *)malloc(sizeField);

    // 2. allocate points
    size_t sizePoints = numPoints*sizeof(float3);
    float3 *h_points = (float3 *)malloc(sizePoints);
    float max_x = -1e10, max_y = -1e10, max_z = -1e10;
    float min_x = 1e10, min_y = 1e10, min_z = 1e10;
    for(unsigned int i = 0; i<numPoints; i++)
    {
        h_points[i].x = cloud.features(0,i); 
        h_points[i].y = cloud.features(1,i); 
        h_points[i].z = cloud.features(2,i); 
        if (cloud.features(0,i) > max_x) max_x = cloud.features(0,i);
        if (cloud.features(1,i) > max_y) max_y = cloud.features(1,i);
        if (cloud.features(2,i) > max_z) max_z = cloud.features(2,i);
        if (cloud.features(0,i) < min_x) min_x = cloud.features(0,i);
        if (cloud.features(1,i) < min_y) min_y = cloud.features(1,i);
        if (cloud.features(2,i) < min_z) min_z = cloud.features(2,i);
    }

    // setup grid
    // dimension extrems
    int hw_size = ceil(hw*sigma/cell_size); // half-window size
    cout << "hw_size is " << hw_size << endl;
    unsigned int grid_dimx = ceil( (max_x - min_x)/ cell_size ) + 2*hw_size +1; // +1 because two ceil function will create 1 additional cell
    unsigned int grid_dimy = ceil( (max_y - min_y)/ cell_size ) + 2*hw_size +1;
    unsigned int grid_dimz = ceil( (max_z - min_z)/ cell_size ) + 2*hw_size +1;
    size_t maxGridSize = grid_dimx * grid_dimy * grid_dimz;
    cout << "@@@@@@@ grid size is " << grid_dimx << " x " << grid_dimy << " x " << grid_dimz << " = " << maxGridSize << "  @@@@@@@@@@" << endl;
    cout << "@@@@@@@ grid ranged from: \n x: [" << min_x << " ," << max_x << "]\n "
        << "y: [" << min_y << " ," << max_y << "]\n "
        << "z: [" << min_z << " ," << max_z << "]\n";
    

    // 3. set log
    size_t sizeLog = numPoints*sizeof(int2);
    int2 * h_log = (int2 *)malloc(sizeLog);
    bzero( h_log, sizeLog);

    // 4. call CUDA
    vector< tuple<float, float, float> > vecDenseCells; // save centers in meter of all meaningful dense cells
    vecDenseCells.reserve(maxGridSize);
    float* stick_saliency_grid = (float *)malloc(maxGridSize*sizeof(float)); // container for resulting saliency
    float* plate_saliency_grid = (float *)malloc(maxGridSize*sizeof(float));
    float* ball_saliency_grid = (float *)malloc(maxGridSize*sizeof(float));
    float3 *grid_normals = (float3 *)malloc(maxGridSize*sizeof(float3)); // container for normals of each grid cell
    cout << "Send to GPU..." << endl;
    CudaVoting::unifiedDenseGridStickVoting(h_points, maxGridSize, hw_size, sigma, cell_size,
                numPoints, grid_dimx, grid_dimy, grid_dimz, 
                min_x, min_y, min_z, 
                stick_saliency_grid, plate_saliency_grid, ball_saliency_grid, grid_normals,
                h_log); // h_pointgrid is left on GPU only.

    // 5. post-processing
    vector<unsigned int> record; // save VALID cells
    record.reserve(maxGridSize);
    // save and grid selection
    for(unsigned int i = 0; i<maxGridSize; i++)
    {
        // save cell definition
        // coord refering origin
        int z = i / (grid_dimx*grid_dimy) - hw_size;
        int xy = i % (grid_dimx*grid_dimy);
        int x = xy % grid_dimx - hw_size;
        int y = xy / grid_dimx - hw_size;

       // cout << "(" << x << ',' << y << ',' << z << ")";


        float xx = x*cell_size + 0.5*cell_size + min_x;
        float yy = y*cell_size + 0.5*cell_size + min_y;
        float zz = z*cell_size + 0.5*cell_size + min_z;

        cout << "(" << xx << ',' << yy << ',' << zz << ")" << endl;

        if (stick_saliency_grid[i] >TH_VALID_STICK && plate_saliency_grid[i]>TH_VALID_PLATE && ball_saliency_grid[i] > TH_VALID_BALL)
        {
            vecDenseCells.push_back( std::make_tuple(xx, yy, zz) );
            record.push_back(i);
        }
    }
    cout << endl;
    cout << "maxGridSize : " << maxGridSize << endl << "realGridSize : " << vecDenseCells.size() << endl;

    // 6. Note: the tensor split is intergtated by unifiedDenseGridStickVoting().
    // Please check the other demo for sparse voting for details if you need the tensor split by GPU.




    // 7. save sparse tensor
    PointMatcher<float>::Matrix grid_points=PM::Matrix::Zero(3, vecDenseCells.size());
    PointMatcher<float>::Matrix stick=PM::Matrix::Zero(1, vecDenseCells.size());
    PointMatcher<float>::Matrix plate=PM::Matrix::Zero(1, vecDenseCells.size());
    PointMatcher<float>::Matrix ball =PM::Matrix::Zero(1, vecDenseCells.size());
    PointMatcher<float>::Matrix normals=PM::Matrix::Zero(3, vecDenseCells.size());
    for(unsigned int i=0; i<vecDenseCells.size(); i++)
    {
        unsigned int v = record[i];
        grid_points.col(i) << std::get<0>(vecDenseCells[i]), std::get<1>(vecDenseCells[i]),std::get<2>(vecDenseCells[i]);
        stick(i) = stick_saliency_grid[v];
        plate(i) = plate_saliency_grid[v]; 
        ball(i)  =  ball_saliency_grid[v];
        normals.col(i) << grid_normals[v].x, grid_normals[v].y, grid_normals[v].z;
    }

    // add grid as DP for "grid"
//    DP::Labels myfeatureLabels;
//    for (size_t i=0 ; i< myfeatureLabels.size(); i++)
//    {
//        myfeatureLabels.push_back(DP::Label("x", 1));
//        myfeatureLabels.push_back(DP::Label("y", 1));
//        myfeatureLabels.push_back(DP::Label("z", 1));
//        myfeatureLabels.push_back(DP::Label("pad", 1));
//    }
//    grid = DP(grid_points, myfeatureLabels);
    grid = DP(grid_points, cloud.featureLabels);
    assert(!grid.featureLabels.empty());
    grid.addDescriptor("stick", stick);
    grid.addDescriptor("plate", plate);
    grid.addDescriptor("ball", ball);
    grid.addDescriptor("normals", normals);

    // 8. clean up
    free(h_fieldarray);
    free(h_points);
    free(h_log);
    free(stick_saliency_grid);
    free(plate_saliency_grid);
    free(ball_saliency_grid);
    free(grid_normals);
}


// Main function supporting the DenseVotingCloudGPU class
int main(int argc, char **argv)
{
	ros::init(argc, argv, "demo_sparse_gpu");
	ros::NodeHandle n;
	DenseVotingCloudGPU gpuDenseVoter(n);
	ros::spin();
	
	return 0;
}
