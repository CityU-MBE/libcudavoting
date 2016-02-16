/*
 * =====================================================================================
 *
 *       Filename:  global.h
 *
 *    Description:  global parameters
 *
 *        Version:  1.0
 *        Created:  03/28/2012 03:02:47 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ming Liu (), ming.liu@mavt.ethz.ch
 *        Company:  ETHZ
 *
 * =====================================================================================
 */

#ifndef GLOBAL_H
#define GLOBAL_H

#include "types.h"

#define MAXNUM_BASIS_PER_CELL 5
#define exact_distance

namespace topomap{
    // check if an element is in a std::map
    template<class mapT, class T>
    bool exists(const T what, const mapT map1)
    {
        typename mapT::const_iterator it = map1.find(what);
        return it!=map1.end();
    }

    // check if an element is in a std::vector ( if in: true )
    template <class T> 
    bool in(const T ele, std::vector<T> vec) {
        if ( vec.size() == 0 )
          return false;

        if (find(vec.begin(), vec.end(), ele)!=vec.end()) {
            return true;
        }
        else {
            return false;
        }
    }



}

#endif
