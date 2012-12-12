/***************************************************************
* File Name: DesignStateBase.h
*
* Description: Base class of design state objects
*
***************************************************************/

#ifndef _DESIGN_STATE_BASE_H_
#define _DESIGN_STATE_BASE_H_


// Local includes
#include "../AnimationModeler/AnimationModelerBase.h"
#include "../DesignObjectHandler.h"
#include "../DesignStateIntersector.h"
#include "../DesignObjectIntersector.h"
#include "../DesignStates/DesignStateParticleSystem.h"

// Open scene graph
#include <osg/NodeCallback>
#include <osg/PositionAttitudeTransform>
#include <osg/Switch>


class DesignStateBase;
typedef std::list<DesignStateBase*> DesignStateList;
typedef std::vector<DesignStateBase*> DesignStateVector;


/***************************************************************
* Class: DesignStateBase
***************************************************************/
class DesignStateBase: public osg::Switch
{
  public:
    DesignStateBase();

    // 'setObjectEnabled' is called when the current state is switched on/off,
	// triggered by left/right button or state toggle button event.
    virtual void setObjectEnabled(bool flag) = 0;

    // switch functions for sub states, triggered by up/down buttons
    virtual void switchToPrevSubState() = 0;
    virtual void switchToNextSubState() = 0;

    // pointer responses and update functions: all functions are stemmed from 'preFrame' function
    virtual void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) = 0;
    virtual bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) = 0;
    virtual bool inputDevReleaseEvent() = 0;
    virtual void update() = 0;
    bool test(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);


    void setParticleSystemPtr(DesignStateParticleSystem *psPtr) { mDSParticleSystemPtr = psPtr; }

    /* Add upper/lower design states, set upper/lower state switch function callbacks:
       These functions are called when design states are being created in 'DesignStateRenderer'
       if any state is using upper or lower states. Callback functions are mapped to static
       state switch functions defined in 'DesignStateRenderer'. */
    void addUpperDesignState(DesignStateBase *dsPtr) { mUpperDSVector.push_back(dsPtr); }
    void addLowerDesignState(DesignStateBase *dsPtr) { mLowerDSVector.push_back(dsPtr); }
    void setUpperStateSwitchCallback(void (*func)(const int &idx)) { mUpperDSSwitchFuncPtr = func; }
    void setLowerStateSwitchCallback(void (*func)(const int &idx)) { mLowerDSSwitchFuncPtr = func; }

    /* Switch between upper/lower design states are called with in derived classes of 'DesignStateBase',
       through which static function in 'DesignStateRenderer' will be called. Within these static
       functions, specific design state pointers with given index can be accessed and activated. */
    void switchToUpperDesignState(const int &idx);
    void switchToLowerDesignState(const int &idx);
    DesignStateBase *getUpperDesignState(const int &idx) { return mUpperDSVector.at(idx); }
    DesignStateBase *getLowerDesignState(const int &idx) { return mLowerDSVector.at(idx); }

    // Lock function: Setup flag that enables/disables switch between different design states
    void setLocked(bool flag) { mLockedFlag = flag; }
    bool isLocked() { return mLockedFlag; }
    bool isEnabled() { return mObjEnabledFlag; }

    virtual void setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, 
        const osg::Vec3 &pointerPos) = 0;

    // static member functions: called within 'DesignStateRenderer' where global information for
	// root groups are written, including group pointer, position and orientation.
    static void setDesignStateRootGroupPtr(osg::Group *designStateRootGroup);
    static void setDesignObjectRootGroupPtr(osg::Group *designObjectRootGroup);
    static void setDesignStateCenterPos(const osg::Vec3 &pos);
    static void setDesignStateFrontVect(const osg::Vec3 &front);

  protected:
    bool mObjEnabledFlag, mDevPressedFlag, mLockedFlag;
    DesignStateParticleSystem *mDSParticleSystemPtr;
    osg::PositionAttitudeTransform *mPATransFwd, *mPATransBwd;

    // for each design state, two intersectors are defined to perform intersection test with
    // 'DesignState' root and 'DesignObject' root respectively
    DSIntersector *mDSIntersector;
    DOIntersector *mDOIntersector;

    // upper & lower design states vectors and switch callback functions
    DesignStateVector mUpperDSVector, mLowerDSVector;
    void (*mUpperDSSwitchFuncPtr)(const int &idx);
    void (*mLowerDSSwitchFuncPtr)(const int &idx);

    // static members: root group of design state & design object, position
    // and front orientation of state sphere
    static osg::Group *gDesignStateRootGroup;
    static osg::Group *gDesignObjectRootGroup;
    static osg::Vec3 gDesignStateCenterPos;
    static osg::Vec3 gDesignStateFrontVect;
    static osg::Matrixd gDesignStateBaseRotMat;
};

#endif

