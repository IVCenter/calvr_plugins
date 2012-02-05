/***************************************************************
* File Name: DSSketchBook.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_SKETCH_BOOK_H_
#define _DS_SKETCH_BOOK_H_


// Local include
#include "DesignStateBase.h"
#include "../VirtualScenicHandler.h"
#include "../AnimationModeler/ANIMSketchBook.h"


/***************************************************************
* Class: DSSketchBook
***************************************************************/
class DSSketchBook: public DesignStateBase
{
  public:
    DSSketchBook();
    ~DSSketchBook();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();

    void setScenicHandlerPtr(VirtualScenicHandler *vsHandlerPtr);

    /* definition of switch lock states */
    enum FlipLockState
    {
	FLIP_UPWARD,
	FLIP_DOWNWARD,
	RELEASED
    };

  protected:
    VirtualScenicHandler *mVirtualScenicHandler;

    bool mReadyToPlaceFlag;
    FlipLockState mFlipLockState;
    osg::AnimationPathCallback *mSignalAnimation;
    osg::Switch *mSignalActivatedSwitch;

    int mPageIdx, mNumPages, mFlipStepsCount;
    CAVEAnimationModeler::ANIMPageEntry **mPageEntryArray;
};


#endif
