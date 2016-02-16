/*
 * =====================================================================================
 *
 *       Filename:  vtktool.h
 *
 *    Description:  vtk related tools
 *                  Cell - grid operation and debugging etc
 *
 *        Version:  1.0
 *        Created:  04/17/2012 10:54:02 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */
#ifndef VTKTOOL_H
#define VTKTOOL_H
#include <string>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include "boost/tuple/tuple.hpp"


namespace topomap{

	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    class VTKTool
    {
        public:
            static void dumpGridCell(const std::string filename, const CellVector cells); 
            static void dumpGridCell(const std::string filename, const CellVector cells,
                        const Matrix stick,
                        const Matrix plate,
                        const Matrix ball
                        ); 
            static void dumpGridCell(const std::string filename, const std::vector< boost::tuple<float, float, float> > vecCell,
                        const float cell_size,
                        const Matrix stick,
                        const Matrix plate,
                        const Matrix ball
                        ); 
    
    };

}


#endif

