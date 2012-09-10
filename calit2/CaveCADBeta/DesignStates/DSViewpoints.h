/***************************************************************
* File Name: DSVirtualSphere.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_VIEWPOINTS_H
#define _DS_VIEWPOINTS_H


// Local include
#include "DesignStateBase.h"
#include "../AnimationModeler/ANIMViewpoints.h"
#include <cvrKernel/PluginHelper.h>


/***************************************************************
* Class: DSVirtualSphere
***************************************************************/
class DSViewpoints: public DesignStateBase
{
  public:
    DSViewpoints();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);// { mDevPressedFlag = true; return false; }
    bool inputDevReleaseEvent();// { mDevPressedFlag = false; return false; }
    void update() {}

    struct Location 
    {
        osg::Vec3 eye;
        osg::Vec3 center;
        osg::Vec3 up;
    };

  protected:
    bool mIsOpen;
    std::vector<osg::PositionAttitudeTransform*> fwdVec, bwdVec;
    std::vector<Location> locations;
    
};


#endif
