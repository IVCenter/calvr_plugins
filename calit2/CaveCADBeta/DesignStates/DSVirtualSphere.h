/***************************************************************
* File Name: DSVirtualSphere.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_VIRTUAL_SPHERE_H
#define _DS_VIRTUAL_SPHERE_H


// Local include
#include "DesignStateBase.h"
#include "../AnimationModeler/ANIMVirtualSphere.h"

#include "DSVirtualEarth.h"
#include "DSParamountSwitch.h"
#include "DSSketchBook.h"
#include <list>

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

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();// {}
    void addChildState(DesignStateBase* ds);

  protected:
    std::list<DesignStateBase*> mChildStates;
    bool mIsOpen;
    std::vector<osg::PositionAttitudeTransform*> fwdVec, bwdVec;
    DesignStateBase *mActiveSubState;
};


#endif
