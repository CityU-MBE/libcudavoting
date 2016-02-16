cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

# Run the CUDA-linking routines
#rosbuild_find_ros_package(cuda)
#message("CUDA Directory" ${cuda_PACKAGE_PATH})

#rosbuild_include(cuda cuda)
#message("CUDA SDK Directory" ${CUDA_SDK_ROOT_DIR})
#include_directories(${cuda_PACKAGE_PATH}/sdk/shared/inc ${cuda_PACKAGE_PATH}/sdk/C/common/inc)



#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

if(CMAKE_COMPILER_IS_GNUCXX)
	execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
	if (GCC_VERSION VERSION_GREATER 4.4 OR GCC_VERSION VERSION_EQUAL 4.4)
		add_definitions(-std=gnu++0x)
	else (GCC_VERSION VERSION_GREATER 4.4 OR GCC_VERSION VERSION_EQUAL 4.4)
		message(SEND_ERROR "You need GCC version 4.4 or greater to compile this library")
	endif (GCC_VERSION VERSION_GREATER 4.4 OR GCC_VERSION VERSION_EQUAL 4.4)
else(CMAKE_COMPILER_IS_GNUCXX)
	message(SEND_ERROR "You need partial C++0x support for N2351, N2672, N1984. Currently this build toolchain supports only GCC >= 4.4. If your compiler supports these specifications, please send us a patch.")
endif(CMAKE_COMPILER_IS_GNUCXX)


rosbuild_find_ros_stack(ethzasl_icp_mapping)
message("ethzasl_icp_mapping_STACK_PATH is " ${ethzasl_icp_mapping_STACK_PATH})

# add pointmatcher
find_path(NABO_INCLUDE_DIR nabo/nabo.h
	${ethzasl_icp_mapping_STACK_PATH}/libnabo/include
)
find_library(NABO_LIBRARY nabo
	${ethzasl_icp_mapping_STACK_PATH}/libnabo/build
)
include_directories(${NABO_INCLUDE_DIR})
message("NABO_INCLUDE_DIR is " ${NABO_INCLUDE_DIR} )

find_path(POINT_MATCHER_INCLUDE_DIR pointmatcher/PointMatcher.h
	${ethzasl_icp_mapping_STACK_PATH}/libpointmatcher/include
)
message("POINT_MATCHER_INCLUDE_DIR is " ${POINT_MATCHER_INCLUDE_DIR})

find_path(POINT_MATCHER_ROS_INCLUDE_DIR pointmatcher_ros/point_cloud.h
	${ethzasl_icp_mapping_STACK_PATH}/libpointmatcher_ros/include
)
message("POINT_MATCHER_ROS_INCLUDE_DIR is " ${POINT_MATCHER_ROS_INCLUDE_DIR})

find_library(POINT_MATCHER_LIBRARY pointmatcher
	${ethzasl_icp_mapping_STACK_PATH}/libpointmatcher/build
)
include_directories(${POINT_MATCHER_INCLUDE_DIR} ${POINT_MATCHER_ROS_INCLUDE_DIR})


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# boost
find_package(Boost COMPONENTS thread filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(BOOST_LIBRARIES ${Boost_LIBRARIES})
LINK_DIRECTORIES( ${LINK_DIRECTORIES} /usr/lib)


# look for yaml-cpp library
find_path(yaml-cpp_INCLUDE_DIRS yaml-cpp/yaml.h
	/usr/local/include
	/usr/include
)
find_library(yaml-cpp_LIBRARIES yaml-cpp PATHS
	/usr/local/lib
	/usr/lib
NO_DEFAULT_PATH)
if (yaml-cpp_INCLUDE_DIRS AND yaml-cpp_LIBRARIES)
	include_directories(${yaml-cpp_INCLUDE_DIRS})
	set(EXTRA_LIBS ${EXTRA_LIBS} ${yaml-cpp_LIBRARIES})
	add_definitions(-DHAVE_YAML_CPP)
endif (yaml-cpp_INCLUDE_DIRS AND yaml-cpp_LIBRARIES)

# look for yaml-cpp library
find_path(eigen_INCLUDE_DIRS Eigen/Eigen
	/usr/local/include
	/usr/include/eigen3
    /usr/include
)
message("eigen_INCLUDE_DIRS is " ${eigen_INCLUDE_DIRS})


if (eigen_INCLUDE_DIRS)
	include_directories(${eigen_INCLUDE_DIRS})
	add_definitions(-DHAVE_EIGEN)
else(eigen_INCLUDE_DIRS)
        message("Eigen Not Found!")
endif (eigen_INCLUDE_DIRS)

find_package(CUDA QUIET)
if(CUDA_FOUND)
#SET(CUDA_NVCC_FLAGS "-arch=sm_30" "-O3" "-use_fast_math")
    SET(CUDA_NVCC_FLAGS "-arch=sm_30" "-O3" "-use_fast_math")
    include_directories( ${CUDA_INCLUDE_DIRS} /usr/local/cuda-5.5/samples/common/inc )
    message("CUDA include directories: ###" ${CUDA_INCLUDE_DIRS} " ###" ${CUDA_INCLUDE_DIRS}/samples/common/inc/  )
    add_definitions(-DUSE_CUDA)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
# Build the kernel and the kernel-aware host code
CUDA_ADD_LIBRARY(CudaVoting src/CudaVoting.cu)


# Build the ROS node and the matrix-adding code that's CUDA-unaware
#rosbuild_add_executable(fake_voting_gpu src/fake_voting_gpu.cpp)
include_directories(/usr/include/eigen3 /usr/local/cuda/include)
rosbuild_add_boost_directories()

#rosbuild_add_executable(fake_voting_gpu src/fake_voting_sparse_gpu.cpp src/tensor_voting.cpp)
#rosbuild_link_boost(fake_voting_gpu  thread system filesystem)
#target_link_libraries(fake_voting_gpu CudaVoting ${POINT_MATCHER_LIBRARY} ${NABO_LIBRARY} ${EXTRA_LIBS} cuda cudart rt ${Boost_LIBRARIES})

#rosbuild_add_executable(demo_sparse_cpu src/demo_sparse_cpu.cpp src/tensor_voting.cpp)
#rosbuild_link_boost(demo_sparse_cpu  thread system filesystem)
#target_link_libraries(demo_sparse_cpu CudaVoting ${POINT_MATCHER_LIBRARY} ${NABO_LIBRARY} ${EXTRA_LIBS} cuda cudart rt ${Boost_LIBRARIES})

# sparse voting demo
rosbuild_add_executable(demo_sparse_gpu src/demo_sparse_gpu.cpp src/tensor_voting.cpp)
rosbuild_link_boost(demo_sparse_gpu  thread system filesystem)
target_link_libraries(demo_sparse_gpu CudaVoting ${POINT_MATCHER_LIBRARY} ${NABO_LIBRARY} ${EXTRA_LIBS} cuda cudart rt ${Boost_LIBRARIES})

# sparse + dense stick voting demo (use unified GPU implementation)
rosbuild_add_executable(demo_dense_gpu src/demo_dense_gpu.cpp src/tensor_voting.cpp)
rosbuild_link_boost(demo_dense_gpu  thread system filesystem)
target_link_libraries(demo_dense_gpu CudaVoting ${POINT_MATCHER_LIBRARY} ${NABO_LIBRARY} ${EXTRA_LIBS} cuda cudart rt ${Boost_LIBRARIES})
