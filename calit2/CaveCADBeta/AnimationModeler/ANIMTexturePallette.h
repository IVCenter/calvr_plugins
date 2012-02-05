/***************************************************************
* File Name: ANIMTexturePallette.h
*
* Description:
*
***************************************************************/

#ifndef _ANIM_TEXTURE_PALLETTE_H_
#define _ANIM_TEXTURE_PALLETTE_H_


// C++
#include <iostream>
#include <list>
#include <string.h>

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
    #define ANIM_TEXTURE_PALLETTE_ENTRY_SPHERE_RADIUS		0.1f
    #define ANIM_TEXTURE_PALLETTE_ANIMATION_TIME		1.0f


    /***************************************************************
    * Class: ANIMTexturePalletteIdleEntry
    ***************************************************************/
    class ANIMTexturePalletteIdleEntry
    {
      public:
	osg::Switch *mEntrySwitch;
	osg::AnimationPathCallback *mFwdAnim, *mBwdAnim;
	osg::Geode *mEntryGeode;
    };


    /***************************************************************
    * Class: ANIMTexturePalletteSelectEntry
    *
    * Eight state animations are defined for each entry:
    * mStateAnim1 / mStateAnim2: Translation from origin to showup
    * pos, zoomed up to half size.
    * mStateAnim3 / mStateAnim4: Zoom between zero and half size at
    * show up position, no translation.
    * mStateAnim5 / mStateAnim6: Zoom between half and full size,
    * translation from showup position to origin.
    * mStateAnim7 / mStateAnim8: Zoom between full size and zero at
    * origin position, no translation.
    *
    ***************************************************************/
    class ANIMTexturePalletteSelectEntry
    {
      public:
	osg::Switch *mEntrySwitch;
	osg::AnimationPathCallback **mStateAnimationArray;
	osg::Geode *mEntryGeode;

	/* sets & gets functions */
	void setDiffuse(const osg::Vec3 &diffuse) { mDiffuse = diffuse; }
	void setSpecular(const osg::Vec3 &specular) { mSpecular = specular; }
	void setTexFilename(const std::string &texfilename) { mTexFilename = texfilename; }
	void setAudioInfo(const std::string &audioinfo) { mAudioInfo = audioinfo; }

	const osg::Vec3 &getDiffuse() { return mDiffuse; }
	const osg::Vec3 &getSpecular() { return mSpecular; }
	const std::string &getTexFilename() { return mTexFilename; }
	const std::string &getAudioInfo() { return mAudioInfo; }

      protected:
	osg::Vec3 mDiffuse, mSpecular;
	std::string mTexFilename;
	std::string mAudioInfo;
    };


    /* function called by 'DSTexturePallette' */
    void ANIMLoadTexturePalletteRoot(osg::PositionAttitudeTransform** xformScaleFwd,
				     osg::PositionAttitudeTransform** xformScaleBwd);
    void ANIMLoadTexturePalletteIdle(osg::Switch **idleStateSwitch, ANIMTexturePalletteIdleEntry **textureStateIdelEntry);
    void ANIMLoadTexturePalletteSelect( osg::Switch **selectStateSwitch, osg::Switch **alphaTurnerSwitch,
					int &numTexs, ANIMTexturePalletteSelectEntry ***textureStatesEntryArray);

    /* local function calls by 'ANIMLoadTexturePalletteSelect' */
    void ANIMCreateTextureEntryGeode(const osg::Vec3 &showUpPos, const osg::Vec3 &diffuse, const osg::Vec3 &specular,
				     const std::string &texfilename, ANIMTexturePalletteSelectEntry **textureEntry);
    void ANIMCreateRandomShowupPosArray(const int &numTexs, osg::Vec3 **showUpPosArray);
};


#endif






