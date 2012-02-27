/***************************************************************
* File Name: CAVEGeodeIcon.h
*
* Description: Derived class from CAVEGeode
*
***************************************************************/

#ifndef _CAVE_GEODE_ICON_H_
#define _CAVE_GEODE_ICON_H_


// C++
#include <iostream>
#include <list>
#include <vector>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>

// local
#include "CAVEGeode.h"
#include "CAVEGeodeShape.h"


class CAVEGeodeIcon;
typedef std::vector<CAVEGeodeIcon*>	CAVEGeodeIconVector;


/***************************************************************
* Class: CAVEGeodeIcon
***************************************************************/
class CAVEGeodeIcon: public CAVEGeode
{
  public:
    CAVEGeodeIcon();
    ~CAVEGeodeIcon();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}

    /* 'gSnappingUnitDist' is the minimum sensible offset distance in CAVE design space. This values does not
        affect the size of virtual icons or surfaces showing by 'DOGeometryCollector' */
    static const float gSnappingUnitDist;

    /* 'gSphereBoundSize' is the maximum radius of sphere bound that contains the first virtual shape selected 
        by 'DOGeometryCollector', all icons will be positioned around this sphere as well */
    static const float gSphereBoundRadius;
};


#endif

