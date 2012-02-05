/***************************************************************
* File Name: DSGeometryCreator.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Nov 10, 2010
*
***************************************************************/
#include "DSGeometryCreator.h"

using namespace std;
using namespace osg;


// Constructor
DSGeometryCreator::DSGeometryCreator(): mShapeSwitchIdx(0), mNumShapeSwitches(0), mDrawingState(IDLE),
					mAudioConfigHandler(NULL)
{
    CAVEAnimationModeler::ANIMLoadGeometryCreator(&mPATransFwd, &mPATransBwd, &mSphereExteriorSwitch, &mSphereExteriorGeode,
							mNumShapeSwitches, &mShapeSwitchEntryArray);
    this->addChild(mPATransFwd);
    this->addChild(mPATransBwd);

    /* create both instances of intersector */
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    mSnapLevelController = new SnapLevelController();

    setAllChildrenOff();
    mDevPressedFlag = false;
}


// Destructor
DSGeometryCreator::~DSGeometryCreator()
{
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSGeometryCreator::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    mDrawingState = IDLE;
    if (flag) setAllChildrenOn();
    if (!mPATransFwd || !mPATransBwd) return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
	setSingleChildOn(0);
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());

	/* load intersection root and targets when state is enabled, no need to change till disabled */
	mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
	mDOIntersector->loadRootTargetNode(gDesignObjectRootGroup, NULL);

	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(0);
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpFwdAnim->reset();
    } else {
	setSingleChildOn(1);
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());

	mDSIntersector->loadRootTargetNode(NULL, NULL);
	mDOIntersector->loadRootTargetNode(NULL, NULL);

	/* turn off all geometry objects associated with DesignObjectHanlder */
	mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
	mDOGeometryCreator->setReferenceAxisMasking(false);
	mDOGeometryCreator->setWireframeActiveID(-1);
	mDOGeometryCreator->setSolidshapeActiveID(-1);
	DrawingStateTransitionHandle(mDrawingState, IDLE);  
    }
    if (animCallback) animCallback->reset();
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSGeometryCreator::switchToPrevSubState()
{
    if (mDrawingState == IDLE)
    {
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(3);
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipDownBwdAnim->reset();

	if (--mShapeSwitchIdx < 0) mShapeSwitchIdx = mNumShapeSwitches - 1;

	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(2);
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpBwdAnim->reset();

	mDOGeometryCreator->setWireframeActiveID(-1);
	mDOGeometryCreator->setResize(0.0f);
    }
    else
    {
	mSnapLevelController->switchToUpperLevel();
	mDOGeometryCreator->setScalePerUnit(  mSnapLevelController->getSnappingLength(),
						mSnapLevelController->getSnappingLengthInfo());
    }
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSGeometryCreator::switchToNextSubState()
{
    if (mDrawingState == IDLE)
    {
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(1);
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipDownFwdAnim->reset();

	if (++mShapeSwitchIdx >= mNumShapeSwitches) mShapeSwitchIdx = 0;

	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(0);
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpFwdAnim->reset();

	mDOGeometryCreator->setWireframeActiveID(-1);
	mDOGeometryCreator->setResize(0.0f);
    }
    else
    {
	mSnapLevelController->switchToLowerLevel();
	mDOGeometryCreator->setScalePerUnit(  mSnapLevelController->getSnappingLength(), 
						mSnapLevelController->getSnappingLengthInfo());
    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSGeometryCreator::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag)
    {
	if (mDrawingState == START_DRAWING)
	{
	    mDOGeometryCreator->setSnapPos(pointerPos);
	    mDOGeometryCreator->updateReferenceAxis();
	}
    }
    if (!mDevPressedFlag)
    {
	if (mDrawingState == READY_TO_DRAW)
	{
	    if (mDOIntersector->test(pointerOrg, pointerPos))
	    {
		mDOGeometryCreator->setReferencePlaneMasking(true, true, true);
		mDOGeometryCreator->updateReferencePlane(mDOIntersector->getWorldHitPosition());
	    }
	    else mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
	}
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSGeometryCreator::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;

    if (mDrawingState == IDLE)
    {
	if (mDSIntersector->test(pointerOrg, pointerPos))
	{
	    mDrawingState = READY_TO_DRAW;
	    DrawingStateTransitionHandle(IDLE, READY_TO_DRAW);

	    /* initialize wireframe geode attached to 'DesignObjectHandler' root */
	    mDOGeometryCreator->setReferenceAxisMasking(false);
	    mDOGeometryCreator->setWireframeActiveID(mShapeSwitchIdx);
	    mDOGeometryCreator->resetWireframeGeodes(gDesignStateCenterPos);
	}

	/* switching to lower state 'DSGeometryEditor' only happens in IDLE state, to be specific, 
	   the state changes only if a CAVEGeodeShape object is intersected */
	else if (mDOIntersector->test(pointerOrg, pointerPos))
	{
	    CAVEGeodeShape *hitCAVEGeode = dynamic_cast <CAVEGeodeShape*>(mDOIntersector->getHitNode());
	    if (hitCAVEGeode)
	    {
		mDOGeometryCollector->setSurfaceCollectionHints(hitCAVEGeode, gDesignStateCenterPos);
		mDOGeometryCollector->toggleCAVEGeodeShape(hitCAVEGeode);

		switchToLowerDesignState(0);
		return false;
	    }
	}
    }
    else if (mDrawingState == READY_TO_DRAW)
    {
	if (mDOIntersector->test(pointerOrg, pointerPos))
	{
	    mDrawingState = START_DRAWING;
	    DrawingStateTransitionHandle(READY_TO_DRAW, START_DRAWING);
 
	    mDOGeometryCreator->setWireframeInitPos(pointerPos);
	    mDOGeometryCreator->setSolidshapeActiveID(mShapeSwitchIdx);
	    mDOGeometryCreator->setPointerDir(pointerPos - pointerOrg);
	    mDOGeometryCreator->setScalePerUnit(mSnapLevelController->getSnappingLength(),
						  mSnapLevelController->getSnappingLengthInfo());
	    mDOGeometryCreator->setSolidshapeInitPos(mDOIntersector->getWorldHitPosition());
	    mDOGeometryCreator->setResize(0.0f);
	    mDOGeometryCreator->setReferenceAxisMasking(true);
	    mDOGeometryCreator->setSnapPos(pointerPos);
	    mDOGeometryCreator->setReferencePlaneMasking(true, true, true);
	} else {
	    mDrawingState = IDLE;
	    DrawingStateTransitionHandle(READY_TO_DRAW, IDLE);

	    mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
	    mDOGeometryCreator->setReferenceAxisMasking(false);
	    mDOGeometryCreator->setWireframeActiveID(-1);
	}
    }

    if (mDrawingState == START_DRAWING) return true;
    else return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSGeometryCreator::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    if (mDrawingState == START_DRAWING)
    {
	mDrawingState = IDLE;
	DrawingStateTransitionHandle(START_DRAWING, IDLE);

	/* finish with Design Object handlers */
	mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
	mDOGeometryCreator->setReferenceAxisMasking(false);
	mDOGeometryCreator->registerSolidShape();
	mDOGeometryCreator->setSolidshapeActiveID(-1);
	mDOGeometryCreator->setWireframeActiveID(-1);

	/* update audio parameters */
	mAudioConfigHandler->updateShapes();

	return true;
    }
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSGeometryCreator::update()
{
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DSGeometryCreator::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
{
    mDOGeometryCollector = designObjectHandler->getDOGeometryCollectorPtr();
    mDOGeometryCreator = designObjectHandler->getDOGeometryCreatorPtr();
}


/***************************************************************
* Function: DrawingStateTransitionHandle()
*
* Take actions on transition between drawing states
*
***************************************************************/
void DSGeometryCreator::DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState)
{
    if (prevState == IDLE && nextState == READY_TO_DRAW)
    {
	mSphereExteriorSwitch->setAllChildrenOff();
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setAllChildrenOff();
	mDSParticleSystemPtr->setEmitterEnabled(true);

	setLocked(true);
    }
    else if (prevState == READY_TO_DRAW && nextState == START_DRAWING)
    {
	mDSParticleSystemPtr->setEmitterEnabled(false);
    }
    else if ((prevState == READY_TO_DRAW && nextState == IDLE) ||
	     (prevState == START_DRAWING && nextState == IDLE) ||
	     (prevState == IDLE && nextState == IDLE))
    {
	mSphereExteriorSwitch->setAllChildrenOn();
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(0);
	mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpFwdAnim->reset();
	mDSParticleSystemPtr->setEmitterEnabled(false);

	setLocked(false);
    }
}






























