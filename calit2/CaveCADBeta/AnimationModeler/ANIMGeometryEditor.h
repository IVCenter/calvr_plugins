/***************************************************************
* File Name: ANIMGeometryEditor.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_GEOMETRY_EDITOR_H_
#define _ANIM_GEOMETRY_EDITOR_H_


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
    #define ANIM_GEOMETRY_EDITOR_TOOLKIT_SHOWUP_TIME		0.5f
    #define ANIM_GEOMETRY_EDITOR_TOOLKIT_SHOWUP_SAMPS		16


    /***************************************************************
    * Class: ANIMIconToolkitSwitchEntry
    *
    * Scene graph hierarchy: mSwitch->PATrans->CAVEGroupIconToolkit 
    *
    ***************************************************************/
    class ANIMIconToolkitSwitchEntry
    {
      public:
	osg::Switch *mSwitch;
	osg::AnimationPathCallback *mFwdAnimCallback, *mBwdAnimCallback;
    };

    /***************************************************************
    * Function: ANIMLoadGeometryEditorIconToolkits()
    *
    * function called by 'DOGeometryEditor'
    *
    ***************************************************************/
    void ANIMLoadGeometryEditorIconToolkits(osg::MatrixTransform **iconToolkitTrans, 
		int &numToolkits, ANIMIconToolkitSwitchEntry ***iconToolkitSwitchEntryArray);
    void ANIMCreateSingleIconToolkitSwitchAnimation(int idx, ANIMIconToolkitSwitchEntry **iconToolkitEntry);
};


#endif


