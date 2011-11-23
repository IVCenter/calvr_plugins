/***************************************************************
* File Name: ANIMVirtualSphere.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_VIRTUAL_SPHERE_H_
#define _ANIM_VIRTUAL_SPHERE_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/BlendFunc>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/PositionAttitudeTransform>
#include <osg/Shader>
#include <osg/ShapeDrawable>
#include <osg/StateAttribute>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Texture2D>
#include <osg/Vec3>
#include <osgDB/ReadFile>

// Local includes
#include "AnimationModelerBase.h"


namespace CAVEAnimationModeler
{
    void ANIMCreateVirtualSphere(osg::PositionAttitudeTransform** xformScaleFwd, 
				 osg::PositionAttitudeTransform** xformScaleBwd);
};


#endif
