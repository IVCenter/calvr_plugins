/***************************************************************
* File Name: DSSketchBook.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Nov 4, 2010
*
***************************************************************/
#include "DSSketchBook.h"

using namespace std;
using namespace osg;


// Constructor
DSSketchBook::DSSketchBook(): mPageIdx(0), mNumPages(1), mFlipStepsCount(0)
{
    CAVEAnimationModeler::ANIMLoadSketchBook(&mPATransFwd, &mPATransBwd, mNumPages, &mPageEntryArray);
    this->addChild(mPATransFwd);
    this->addChild(mPATransBwd);

    setAllChildrenOff();
    mReadyToPlaceFlag = false;
    mFlipLockState = RELEASED;
    mSignalAnimation = NULL;
    mSignalActivatedSwitch = NULL;

    /* create instance of intersector */
    mDSIntersector = new DSIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);

    mDevPressedFlag = false;
}


// Destructor
DSSketchBook::~DSSketchBook()
{
    for (int i = 0; i < mNumPages; i++) delete mPageEntryArray[i];
    delete mPageEntryArray;
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSSketchBook::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    if (flag) setAllChildrenOn();
    if (!mPATransFwd || !mPATransBwd) return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
	setSingleChildOn(0);
	mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mPageEntryArray[mPageIdx]->mPageGeode);
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
    } else {
	setSingleChildOn(1);
	mDSIntersector->loadRootTargetNode(NULL, NULL);
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());
    }
    if (animCallback) animCallback->reset();
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSSketchBook::switchToPrevSubState()
{
    if (mFlipLockState != RELEASED) return;
    else mFlipLockState = FLIP_UPWARD;

    int idxNext = mPageIdx + 1;
    if (idxNext >= mNumPages) idxNext = 0;
    mSignalActivatedSwitch = mPageEntryArray[idxNext]->mSwitch;

    mPageEntryArray[mPageIdx]->mSwitch->setSingleChildOn(0);
    mPageEntryArray[mPageIdx]->mFlipUpAnim->reset();
    mSignalAnimation = mPageEntryArray[mPageIdx]->mFlipUpAnim;

    if (--mPageIdx < 0) mPageIdx = mNumPages - 1;

    mPageEntryArray[mPageIdx]->mSwitch->setSingleChildOn(1);
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mPageEntryArray[mPageIdx]->mPageGeode);
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSSketchBook::switchToNextSubState()
{
    if (mFlipLockState != RELEASED) return;
    else mFlipLockState = FLIP_DOWNWARD;

    mSignalActivatedSwitch = mPageEntryArray[mPageIdx]->mSwitch;

    if (++mPageIdx >= mNumPages) mPageIdx = 0;

    int idxNext = mPageIdx + 1;
    if (idxNext >= mNumPages) idxNext = 0;
    mPageEntryArray[idxNext]->mSwitch->setSingleChildOn(0);

    mPageEntryArray[mPageIdx]->mSwitch->setSingleChildOn(1);
    mPageEntryArray[mPageIdx]->mFlipDownAnim->reset();
    mSignalAnimation = mPageEntryArray[mPageIdx]->mFlipDownAnim;
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mPageEntryArray[mPageIdx]->mPageGeode);
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSSketchBook::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag)
    {
	if (!mReadyToPlaceFlag)
	{
	    mReadyToPlaceFlag = mDSIntersector->test(pointerOrg, pointerPos);
	    if (mReadyToPlaceFlag) 
		mVirtualScenicHandler->setFloorplanPreviewHighlight(true, mPageEntryArray[mPageIdx]->mPageGeode);
	}
	else 
	{
	    if (!mDSIntersector->test(pointerOrg, pointerPos))
	    {
		mVirtualScenicHandler->switchFloorplan(mPageIdx, VirtualScenicHandler::TRANSPARENT);
	    } else {
		mVirtualScenicHandler->switchFloorplan(mPageIdx, VirtualScenicHandler::INVISIBLE);
	    }
	}
    }
    else if (!mDevPressedFlag)
    {
	if (mReadyToPlaceFlag)
	{
	    if (!mDSIntersector->test(pointerOrg, pointerPos))
	    {
		mVirtualScenicHandler->switchFloorplan(mPageIdx, VirtualScenicHandler::SOLID);
	    }
	}
	mReadyToPlaceFlag = false;
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSSketchBook::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSSketchBook::inputDevReleaseEvent()
{
    mDevPressedFlag = false;
    mVirtualScenicHandler->setFloorplanPreviewHighlight(false, mPageEntryArray[mPageIdx]->mPageGeode);
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSSketchBook::update()
{
    if (mFlipLockState == RELEASED) return;
    if (mSignalAnimation && mSignalActivatedSwitch)
    {
	float animTime = mSignalAnimation->getAnimationTime();
	if (animTime >= 0.9)
	{
	    mSignalActivatedSwitch->setAllChildrenOff();
	    mFlipLockState = RELEASED;
	    mSignalAnimation = NULL;
	    mSignalActivatedSwitch = NULL;
	}
    }
}


/***************************************************************
* Function: setScenicHandlerPtr()
***************************************************************/
void DSSketchBook::setScenicHandlerPtr(VirtualScenicHandler *vsHandlerPtr)
{
    mVirtualScenicHandler = vsHandlerPtr;
    mVirtualScenicHandler->createFloorplanGeometry(mNumPages, mPageEntryArray);
}



















