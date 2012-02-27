/***************************************************************
* File Name: NavigationHandler.h
*
* Class Name: NavigationHandler
*
***************************************************************/

#ifndef _NAVIGATION_HANDLER_H_
#define _NAVIGATION_HANDLER_H_


// C++
#include <iostream>

// open scene graph
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrixd>

// CalVR plugin support
#include <kernel/PluginHelper.h>


/***************************************************************
* Class: NavigationHandler
***************************************************************/
class NavigationHandler
{
  public:
    NavigationHandler();

    /* button type enumerations */
    enum ButtonType
    {
		STILL = 0x00,
		LEFT = 0x01,
		RIGHT = 0x02,
		UP = 0x04,
		DOWN = 0x08
    };

    void setEnabled(bool flag) { mFlagEnabled = flag; }

    void updateNaviStates(const float &scale, const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);
    void updateButtonStates();
    void updateKeys(const int &keySym, bool pressFlag);
    void updateXformMat();

  protected:
    bool mFlagEnabled;
    float mScale, mMovSpeed, mRotSpeed;
    osg::Vec3 mViewDir, mViewPos;

    ButtonType mButtonType;

    static float gMovSpeedUnit;
    static float gRotSpeedUnit;
};


#endif
