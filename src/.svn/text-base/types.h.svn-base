/*
 * =====================================================================================
 *
 *       Filename:  types.h
 *
 *    Description:  type defines
 *
 *        Version:  1.0
 *        Created:  04/14/2012 11:06:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */
#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include <map>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
namespace  topomap{

            struct Cell;

            // type defines
            typedef boost::tuple< float, float, float > NormalType;
            typedef boost::tuple< float, float, float > TensorEigenValuesType;
            typedef std::vector<Cell> CellVector;
            typedef std::vector<NormalType> NormalsType;
            typedef boost::tuple< int, int, int > PoseIDType;
            typedef std::vector<PoseIDType> LinkedObstacleType;
            typedef std::vector<PoseIDType> FreeCellsType;
            typedef std::vector<PoseIDType> VoronoiGraphCellsType;
            typedef std::vector<LinkedObstacleType> LinkedObstaclesType; // not good naming
            typedef std::map<PoseIDType, CellVector::iterator>  HandlerType;


}

#endif

