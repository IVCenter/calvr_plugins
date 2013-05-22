/***************************************************************
* File Name: DSLights.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_LIGHTS_H
#define _DS_LIGHTS_H


// Local include
#include "DesignStateBase.h"
#include "../AudioConfigHandler.h"
#include "../SnapLevelController.h"
#include "../DesignObjects/DOGeometryCreator.h"
#include "../AnimationModeler/ANIMLights.h"


/***************************************************************
* Class: DSLights
***************************************************************/
class DSLights: public DesignStateBase
{
  public:
    DSLights();
    ~DSLights();

    // virtual functions inherited from base class
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();
    void setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);

    // definition of drawing states
    enum DrawingState
    {
        IDLE,
        PLACE_OBJECT,
    };

    /* link 'DesignObjectHandler->DOGeometryCreator' to 'DSObjectPlacer'
       class 'DSObjectPlacer' listens to input devices and translate the
       input commands to geometry actions, communicate with 'DOGeometryCreator'
       to make these actions happen.
    */
    void setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler);

  protected:
    std::vector<osg::PositionAttitudeTransform*> mFwdVec, mBwdVec;
    
    // Menu geometry
    osg::Switch *mSphereExteriorSwitch;		// switch on/off container exterior sphere
    osg::Geode *mSphereExteriorGeode;		// exterior geode target for intersection test
    osg::Geode *mHighlightGeode;
    osg::ShapeDrawable *mSD;
    int mShapeSwitchIdx, mNumShapeSwitches;
    bool mIsOpen, mIsHighlighted;
    std::vector<osg::PositionAttitudeTransform*> fwdVec, bwdVec;

    CAVEGeodeShape *prevGeode;

    DrawingState mDrawingState;
    osg::PositionAttitudeTransform *_activeObject;

    // interfaces to design object (DO) controllers
    DOGeometryCollector *mDOGeometryCollector;
    DOGeometryCreator *mDOGeometryCreator;
 
    void DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState);
};


#endif

