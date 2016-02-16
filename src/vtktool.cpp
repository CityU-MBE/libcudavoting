/*
 * =====================================================================================
 *
 *       Filename:  vtktool.cpp
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
#include "types.h"
#include "occupancygrid.h"
#include "vtktool.h"

#include <iostream>
#include <fstream>

using namespace std;
namespace topomap{

        void VTKTool::dumpGridCell(const std::string filename, const std::vector< boost::tuple<float, float, float> > cells,
                    const float cell_size,
                    const Matrix stick,
                    const Matrix plate,
                    const Matrix ball
                    )
        {
               /*
                # vtk DataFile Version 3.0
                My unstructured Grid Example 2
                ASCII
                DATASET UNSTRUCTURED_GRID
                POINTS 27 float
                0 0 0 
                1 0 0 
                2 0 0 
                0 1 0  
                CELLS 11 60 // 11 groups, 60 point-entries related in all
                8 0 1 .. // 8 element, 9 entries
                4 1 2 3 4 // 4 element, 5 entries
                CELL_TYPE 11
                11 // voxel, need 8 defining elements
                11
                ..
                */
            ofstream out(filename.c_str());
            const unsigned int num(cells.size()); // number of cells

            // 1. Header
            out << "# vtk DataFile Version 3.0\n";
            out << "3D Occupancy Grid used by topomap\n";
            out << "ASCII\n";
            out << "DATASET UNSTRUCTURED_GRID\n";

            // 2. Tokens defining points
            out << "POINTS " << num*8 << " float" << endl; // each token has 8 defining points
            for (unsigned int i = 0; i<num; i++)
            {
                // token center positions
                float cx = cells[i].get<0>();
                float cy = cells[i].get<1>();
                float cz = cells[i].get<2>();
                // offset, half window
                float hx = cell_size/2;
                float hy = cell_size/2;
                float hz = cell_size/2;
                // find 8 defining points of each token
                // page 9 of : http://www.vtk.org/VTK/img/file-formats.pdf
                // 0
                out << cx-hx << ' ' << cy-hy << ' ' << cz-hz << endl;
                // 1
                out << cx+hx << ' ' << cy-hy << ' ' << cz-hz << endl;
                // 2
                out << cx-hx << ' ' << cy+hy << ' ' << cz-hz << endl;
                // 3
                out << cx+hx << ' ' << cy+hy << ' ' << cz-hz << endl;
                // 4
                out << cx-hx << ' ' << cy-hy << ' ' << cz+hz << endl;
                // 5
                out << cx+hx << ' ' << cy-hy << ' ' << cz+hz << endl;
                // 6
                out << cx-hx << ' ' << cy+hy << ' ' << cz+hz << endl;
                // 7
                out << cx+hx << ' ' << cy+hy << ' ' << cz+hz << endl;
            }
// 3. Indexing defining points for each cell
            out << "CELLS " << num << ' ' << num*9 << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << "8 " << i*8 << ' ' << i*8+1 << ' ' << i*8+2 << ' ' << i*8+3 << ' ' 
                    << i*8+4 << ' ' << i*8+5 << ' ' << i*8+6 << ' ' << i*8+7 << endl;
            }

            // 4. cell display type, here is VTK_VOXEL: 11
            out << "CELL_TYPES " << num << endl;
            for (unsigned int i = 0; i<num; i++)
                out << "11\n";

            // 5. write tensors: stick plate ball
            out << "POINT_DATA "<< num*8 << endl;
            out << "SCALARS stick float 1" << endl;
            out << "LOOKUP_TABLE default" << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
            }

            out << "SCALARS plate float 1" << endl;
            out << "LOOKUP_TABLE default" << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
            }

            out << "SCALARS ball float 1" << endl;
            out << "LOOKUP_TABLE default" << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
            }

            out.close();
        }
    
        void VTKTool::dumpGridCell(const std::string filename, const CellVector cells,
                        const Matrix stick,
                        const Matrix plate,
                        const Matrix ball
                        )
        {
            /*
                # vtk DataFile Version 3.0
                My unstructured Grid Example 2
                ASCII
                DATASET UNSTRUCTURED_GRID
                POINTS 27 float
                0 0 0 
                1 0 0 
                2 0 0 
                0 1 0  
                CELLS 11 60 // 11 groups, 60 point-entries related in all
                8 0 1 .. // 8 element, 9 entries
                4 1 2 3 4 // 4 element, 5 entries
                CELL_TYPE 11
                11 // voxel, need 8 defining elements
                11
                ..
                */
            ofstream out(filename.c_str());
            const unsigned int num(cells.size()); // number of cells

            // 1. Header
            out << "# vtk DataFile Version 3.0\n";
            out << "3D Occupancy Grid used by topomap\n";
            out << "ASCII\n";
            out << "DATASET UNSTRUCTURED_GRID\n";

            // 2. Tokens defining points
            out << "POINTS " << num*8 << " float" << endl; // each token has 8 defining points
            for (unsigned int i = 0; i<num; i++)
            {
                // token center positions
                float cx = cells[i].pose.get<0>();
                float cy = cells[i].pose.get<1>();
                float cz = cells[i].pose.get<2>();
                // offset, half window
                float hx = cells[i].len.get<0>()/2;
                float hy = cells[i].len.get<1>()/2;
                float hz = cells[i].len.get<2>()/2;
                // find 8 defining points of each token
                // page 9 of : http://www.vtk.org/VTK/img/file-formats.pdf
                // 0
                out << cx-hx << ' ' << cy-hy << ' ' << cz-hz << endl;
                // 1
                out << cx+hx << ' ' << cy-hy << ' ' << cz-hz << endl;
                // 2
                out << cx-hx << ' ' << cy+hy << ' ' << cz-hz << endl;
                // 3
                out << cx+hx << ' ' << cy+hy << ' ' << cz-hz << endl;
                // 4
                out << cx-hx << ' ' << cy-hy << ' ' << cz+hz << endl;
                // 5
                out << cx+hx << ' ' << cy-hy << ' ' << cz+hz << endl;
                // 6
                out << cx-hx << ' ' << cy+hy << ' ' << cz+hz << endl;
                // 7
                out << cx+hx << ' ' << cy+hy << ' ' << cz+hz << endl;
            }

            // 3. Indexing defining points for each cell
            out << "CELLS " << num << ' ' << num*9 << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << "8 " << i*8 << ' ' << i*8+1 << ' ' << i*8+2 << ' ' << i*8+3 << ' ' 
                    << i*8+4 << ' ' << i*8+5 << ' ' << i*8+6 << ' ' << i*8+7 << endl;
            }

            // 4. cell display type, here is VTK_VOXEL: 11
            out << "CELL_TYPES " << num << endl;
            for (unsigned int i = 0; i<num; i++)
                out << "11\n";

            // 5. write tensors: stick plate ball
            out << "POINT_DATA "<< num*8 << endl;
            out << "SCALARS stick float 1" << endl;
            out << "LOOKUP_TABLE default" << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
                out << stick(i) << endl;
            }

            out << "SCALARS plate float 1" << endl;
            out << "LOOKUP_TABLE default" << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
                out << plate(i) << endl;
            }

            out << "SCALARS ball float 1" << endl;
            out << "LOOKUP_TABLE default" << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
                out << ball(i) << endl;
            }

            out.close();
        
        }

        void VTKTool::dumpGridCell(const string filename, const CellVector cells)
        {
            /*
                # vtk DataFile Version 3.0
                My unstructured Grid Example 2
                ASCII
                DATASET UNSTRUCTURED_GRID
                POINTS 27 float
                0 0 0 
                1 0 0 
                2 0 0 
                0 1 0  
                CELLS 11 60 // 11 groups, 60 point-entries related in all
                8 0 1 .. // 8 element, 9 entries
                4 1 2 3 4 // 4 element, 5 entries
                CELL_TYPE 11
                11 // voxel, need 8 defining elements
                11
                ..
                */
            ofstream out(filename.c_str());
            const unsigned int num(cells.size()); // number of cells

            // 1. Header
            out << "# vtk DataFile Version 3.0\n";
            out << "3D Occupancy Grid used by topomap\n";
            out << "ASCII\n";
            out << "DATASET UNSTRUCTURED_GRID\n";

            // 2. Tokens defining points
            out << "POINTS " << num*8 << " float" << endl; // each token has 8 defining points
            for (unsigned int i = 0; i<num; i++)
            {
                // token center positions
                float cx = cells[i].pose.get<0>();
                float cy = cells[i].pose.get<1>();
                float cz = cells[i].pose.get<2>();
                // offset, half window
                float hx = cells[i].len.get<0>()/2;
                float hy = cells[i].len.get<1>()/2;
                float hz = cells[i].len.get<2>()/2;
                // find 8 defining points of each token
                // page 9 of : http://www.vtk.org/VTK/img/file-formats.pdf
                // 0
                out << cx-hx << ' ' << cy-hy << ' ' << cz-hz << endl;
                // 1
                out << cx+hx << ' ' << cy-hy << ' ' << cz-hz << endl;
                // 2
                out << cx-hx << ' ' << cy+hy << ' ' << cz-hz << endl;
                // 3
                out << cx+hx << ' ' << cy+hy << ' ' << cz-hz << endl;
                // 4
                out << cx-hx << ' ' << cy-hy << ' ' << cz+hz << endl;
                // 5
                out << cx+hx << ' ' << cy-hy << ' ' << cz+hz << endl;
                // 6
                out << cx-hx << ' ' << cy+hy << ' ' << cz+hz << endl;
                // 7
                out << cx+hx << ' ' << cy+hy << ' ' << cz+hz << endl;
            }

            // 3. Indexing defining points for each cell
            out << "CELLS " << num << ' ' << num*9 << endl;
            for (unsigned int i = 0; i<num; i++)
            {
                out << "8 " << i*8 << ' ' << i*8+1 << ' ' << i*8+2 << ' ' << i*8+3 << ' ' 
                    << i*8+4 << ' ' << i*8+5 << ' ' << i*8+6 << ' ' << i*8+7 << endl;
            }

            // 4. cell display type, here is VTK_VOXEL: 11
            out << "CELL_TYPES " << num << endl;
            for (unsigned int i = 0; i<num; i++)
                out << "11\n";

            out.close();
        }

}

