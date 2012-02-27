/***************************************************************
* File Name: ANIMSketchBook.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_SKETCH_BOOK_H_
#define _ANIM_SKETCH_BOOK_H_


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
    * Class: ANIMPageEntry
    ***************************************************************/
    class ANIMPageEntry
    {
      public:
	osg::Switch *mSwitch;
	osg::AnimationPathCallback *mFlipUpAnim, *mFlipDownAnim;
	osg::Geode *mPageGeode;
	std::string mTexFilename;
	float mLength, mWidth, mAlti;
    };

    void ANIMLoadSketchBook(osg::PositionAttitudeTransform** xformScaleFwd, 
			    osg::PositionAttitudeTransform** xformScaleBwd,
			    int &numPages, ANIMPageEntry ***pageEntryArray);
    void ANIMCreateSinglePageGeodeAnimation(const std::string& texfilename,
					    osg::Geode **flipUpGeode, osg::Geode **flipDownGeode, 
					    osg::AnimationPathCallback **flipUpCallback,
					    osg::AnimationPathCallback **flipDownCallback);
};


#endif
