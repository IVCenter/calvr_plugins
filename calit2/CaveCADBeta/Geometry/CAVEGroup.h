/***************************************************************
* File Name: CAVEGroup.h
*
* Class Name: CAVEGroup
*
***************************************************************/

#ifndef _CAVE_GROUP_H_
#define _CAVE_GROUP_H_


// C++
#include <iostream>
#include <list>
#include <string>

// Open scene graph
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>

// Local include
#include "CAVEGeode.h"


/***************************************************************
* Class: CAVEGroup
***************************************************************/
class CAVEGroup: public osg::Group
{
  public:
    CAVEGroup() {}
    ~CAVEGroup() {}
};


#endif
