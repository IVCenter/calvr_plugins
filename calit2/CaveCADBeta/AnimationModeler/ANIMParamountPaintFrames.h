/***************************************************************
* File Name: ANIMParamountPaintFrames.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_PARAMOUNT_FRAMES_H_
#define _ANIM_PARAMOUNT_FRAMES_H_


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
    /***************************************************************
    * Class: ANIMParamountSwitchEntry
    ***************************************************************/
    class ANIMParamountSwitchEntry
    {
      public:
	osg::MatrixTransform *mMatrixTrans;
	osg::Switch *mSwitch;
	osg::AnimationPathCallback *mZoomInAnim, *mZoomOutAnim;
	osg::Geode *mPaintGeode;
	std::string mTexFilename;
    };

    void ANIMLoadParamountPaintFrames(	osg::PositionAttitudeTransform** xformScaleFwd, 
					osg::PositionAttitudeTransform** xformScaleBwd,
					int &numParas, float &paraswitchRadius, 
					ANIMParamountSwitchEntry ***paraEntryArray);
    void ANIMCreateParamountPaintFrameAnimation(osg::AnimationPathCallback **zoomInCallback,
						osg::AnimationPathCallback **zoomOutCallback);
    osg::Geode *ANIMCreateParamountPaintGeode(const std::string &texfilename);
};


#endif


