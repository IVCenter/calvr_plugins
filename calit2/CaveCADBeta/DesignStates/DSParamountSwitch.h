/***************************************************************
* File Name: DSParamountSwitch.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_PARAMOUNT_SWITCH_H_
#define _DS_PARAMOUNT_SWITCH_H_


// Local include
#include "DesignStateBase.h"
#include "../VirtualScenicHandler.h"
#include "../AnimationModeler/ANIMParamountPaintFrames.h"


/***************************************************************
* Class: DSParamountSwitch
***************************************************************/
class DSParamountSwitch: public DesignStateBase
{
  public:
    DSParamountSwitch();
    ~DSParamountSwitch();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();

    void setScenicHandlerPtr(VirtualScenicHandler *vsHandlerPtr) { mVirtualScenicHandler = vsHandlerPtr; }

    /* definition of switch lock states */
    enum SwitchLockState
    {
	ROTATE_FORWARD,
	ROTATE_BACKWARD,
	RELEASED
    };

  protected:
    VirtualScenicHandler *mVirtualScenicHandler;

    bool mSwitchReadyFlag;
    SwitchLockState mSwitchLockState;
    int mStateParaIdx, mNumParas, mRotStepsCount;
    float mSwitchRadius;
    CAVEAnimationModeler::ANIMParamountSwitchEntry **mParaEntryArray;
};


#endif
