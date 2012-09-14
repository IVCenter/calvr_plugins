/***************************************************************
* File Name: DSGeometryCreator.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_GEOMETRY_CREATOR_H_
#define _DS_GEOMETRY_CREATOR_H_


// Local include
#include "DesignStateBase.h"
#include "../AudioConfigHandler.h"
#include "../SnapLevelController.h"
#include "../DesignObjects/DOGeometryCreator.h"
#include "../AnimationModeler/ANIMGeometryCreator.h"


/***************************************************************
* Class: DSGeometryCreator
***************************************************************/
class DSGeometryCreator: public DesignStateBase
{
  public:
    DSGeometryCreator();
    ~DSGeometryCreator();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();

    /* definition of drawing states */
    enum DrawingState
    {
        IDLE,
        READY_TO_DRAW,
        START_DRAWING
    };

    /* link 'DesignObjectHandler->DOGeometryCreator' to 'DSGeometryCreator'
       class 'DSGeometryCreator' listens to input devices and translate the
       input commands to geometry actions, communicate with 'DOGeometryCreator'
       to make these actions happen.
    */
    void setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler);
    void setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler) { mAudioConfigHandler = audioConfigHandler; }

  protected:
    osg::Switch *mSphereExteriorSwitch;		// switch on/off container exterior sphere
    osg::Geode *mSphereExteriorGeode;		// exterior geode target for intersection test
    int mShapeSwitchIdx, mNumShapeSwitches;
    CAVEAnimationModeler::ANIMShapeSwitchEntry **mShapeSwitchEntryArray;
    bool mIsOpen;
    std::vector<osg::PositionAttitudeTransform*> fwdVec, bwdVec;

    CAVEGeodeShape *prevGeode;

    DrawingState mDrawingState;
    SnapLevelController *mSnapLevelController;
    AudioConfigHandler *mAudioConfigHandler;

    /* interfaces to design object (DO) controllers */
    DOGeometryCollector *mDOGeometryCollector;
    DOGeometryCreator *mDOGeometryCreator;
 
    void DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState);
};


#endif

