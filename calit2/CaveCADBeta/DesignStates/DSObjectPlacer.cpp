/***************************************************************
* File Name: DSObjectPlacer.cpp
*
* Description: 
*
* Written by Cathy Hughes 25 Sept 2012
*
***************************************************************/
#include "DSObjectPlacer.h"

using namespace std;
using namespace osg;


// Constructor
DSObjectPlacer::DSObjectPlacer()
{
    CAVEAnimationModeler::ANIMCreateObjectPlacer(&mFwdVec, &mBwdVec);
    for (int i = 0; i < mFwdVec.size(); ++i)
    {
        this->addChild(mFwdVec[i]);
        this->addChild(mBwdVec[i]);
    }

    // create both instances of intersector
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    setAllChildrenOff();
    mDevPressedFlag = false;

    prevGeode = NULL;
}


// Destructor
DSObjectPlacer::~DSObjectPlacer()
{

}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSObjectPlacer::setObjectEnabled(bool flag)
{
    mDrawingState = IDLE;

    mObjEnabledFlag = flag;
    AnimationPathCallback* animCallback = NULL;
    setAllChildrenOff();
    if (flag && !mIsOpen) // open menu
    {
        setSingleChildOn(0);
        for (int i = 0; i < mFwdVec.size(); ++i)
        {
            setChildValue(mFwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (mFwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[i]->getChild(0));

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = true;
    }
    else // close menu
    {
        setSingleChildOn(0);
        for (int i = 1; i < mBwdVec.size(); ++i)
        {
            setChildValue(mBwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (mBwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mBwdVec[i]);

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = false;
    }

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSObjectPlacer::switchToPrevSubState()
{
    if (mDrawingState == IDLE)
    {

    }
    else
    {

    }
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSObjectPlacer::switchToNextSubState()
{
    if (mDrawingState == IDLE)
    {

    }
    else
    {

    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSObjectPlacer::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag)
    {
        if (mDrawingState == START_DRAWING)
        {

        }
    }
    if (!mDevPressedFlag)
    {

    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSObjectPlacer::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;

    if (mDrawingState == IDLE)
    {

    }

    else if (mDrawingState == READY_TO_DRAW)
    {

    }

    if (mDrawingState == START_DRAWING) 
        return true;
    else 
        return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSObjectPlacer::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    if (mDrawingState == START_DRAWING)
    {

        return true;
    }
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSObjectPlacer::update()
{
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DSObjectPlacer::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
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
void DSObjectPlacer::DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState)
{
    if (prevState == IDLE && nextState == READY_TO_DRAW)
    {

    }

    else if (prevState == READY_TO_DRAW && nextState == START_DRAWING)
    {

    }

    else if ((prevState == READY_TO_DRAW && nextState == IDLE) ||
	         (prevState == START_DRAWING && nextState == IDLE) ||
	         (prevState == IDLE          && nextState == IDLE))
    {

    }
}


void DSObjectPlacer::setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) 
{
    int idx = -1;
    mIsHighlighted = isHighlighted;


    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
}
 
