/***************************************************************
* File Name: DSVirtualSphere.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_VIRTUAL_SPHERE_H_
#define _DS_VIRTUAL_SPHERE_H_


// Local include
#include "DesignStateBase.h"
#include "../AnimationModeler/ANIMVirtualSphere.h"


/***************************************************************
* Class: DSVirtualSphere
***************************************************************/
class DSVirtualSphere: public DesignStateBase
{
  public:
    DSVirtualSphere();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) {}
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) { mDevPressedFlag = true; return false; }
    bool inputDevReleaseEvent() { mDevPressedFlag = false; return false; }
    void update() {}
};


#endif
