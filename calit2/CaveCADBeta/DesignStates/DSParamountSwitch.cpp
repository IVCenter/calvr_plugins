/***************************************************************
* File Name: DSParamountSwitch.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Oct 28, 2010
*
***************************************************************/
#include "DSParamountSwitch.h"

using namespace std;
using namespace osg;


// Constructor
DSParamountSwitch::DSParamountSwitch(): mStateParaIdx(0), mNumParas(1), mRotStepsCount(0), mSwitchRadius(1.0f) 
{
    CAVEAnimationModeler::ANIMLoadParamountPaintFrames(&mPATransFwd, &mPATransBwd, 
					mNumParas, mSwitchRadius, &mParaEntryArray);
    this->addChild(mPATransFwd);
    this->addChild(mPATransBwd);

    setAllChildrenOff();
    mSwitchReadyFlag = false;
    mSwitchLockState = RELEASED;

    /* create instance of intersector */
    mDSIntersector = new DSIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);

    mDevPressedFlag = false;
}


// Destructor
DSParamountSwitch::~DSParamountSwitch()
{
    for (int i = 0; i < mNumParas; i++) 
    {
        delete mParaEntryArray[i];
    }
    delete mParaEntryArray;
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSParamountSwitch::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    if (flag) 
        setAllChildrenOn();
    if (!mPATransFwd || !mPATransBwd) 
        return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
        setSingleChildOn(0);
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mParaEntryArray[mStateParaIdx]->mPaintGeode);
        animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
    } 
    else 
    {
    /*    setSingleChildOn(1);
        mDSIntersector->loadRootTargetNode(NULL, NULL);
        animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());
    */
    }

    if (animCallback) 
        animCallback->reset();
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSParamountSwitch::switchToPrevSubState()
{
    if (mSwitchLockState != RELEASED) 
        return;
    else 
        mSwitchLockState = ROTATE_BACKWARD;

    mParaEntryArray[mStateParaIdx]->mSwitch->setSingleChildOn(1);
    mParaEntryArray[mStateParaIdx]->mZoomOutAnim->reset();

    if (--mStateParaIdx < 0) 
        mStateParaIdx = mNumParas - 1;

    mParaEntryArray[mStateParaIdx]->mSwitch->setSingleChildOn(0);
    mParaEntryArray[mStateParaIdx]->mZoomInAnim->reset();

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mParaEntryArray[mStateParaIdx]->mPaintGeode);
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSParamountSwitch::switchToNextSubState()
{
    if (mSwitchLockState != RELEASED) 
        return;
    else 
        mSwitchLockState = ROTATE_FORWARD;

    mParaEntryArray[mStateParaIdx]->mSwitch->setSingleChildOn(1);
    mParaEntryArray[mStateParaIdx]->mZoomOutAnim->reset();

    if (++mStateParaIdx >= mNumParas) 
        mStateParaIdx = 0;

    mParaEntryArray[mStateParaIdx]->mSwitch->setSingleChildOn(0);
    mParaEntryArray[mStateParaIdx]->mZoomInAnim->reset();

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mParaEntryArray[mStateParaIdx]->mPaintGeode);
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSParamountSwitch::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    if (mDevPressedFlag)
    {
        if (!mSwitchReadyFlag)
        {
            mSwitchReadyFlag = mDSIntersector->test(pointerOrg, pointerPos);
            if (mSwitchReadyFlag)
            {
                mVirtualScenicHandler->setVSParamountPreviewHighlight(true, 
                    mParaEntryArray[mStateParaIdx]->mPaintGeode);
            }
        }
        else 
        {
            mVirtualScenicHandler->setSkyMaskingColorEnabled(!mDSIntersector->test(pointerOrg, pointerPos));
        }
    }
    else if (!mDevPressedFlag)
    {
        if (mSwitchReadyFlag)
        {
            if (!mDSIntersector->test(pointerOrg, pointerPos))
            {
                mVirtualScenicHandler->switchVSParamount(mParaEntryArray[mStateParaIdx]->mTexFilename);
                mVirtualScenicHandler->setSkyMaskingColorEnabled(false);
            }
        }
        mSwitchReadyFlag = false;
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSParamountSwitch::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSParamountSwitch::inputDevReleaseEvent()
{
    mDevPressedFlag = false;
    mVirtualScenicHandler->setVSParamountPreviewHighlight(false, mParaEntryArray[mStateParaIdx]->mPaintGeode);
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSParamountSwitch::update()
{
    if (mSwitchLockState == RELEASED) return;

    /* release rotation lock flag */
    if (++mRotStepsCount > ANIM_PARA_PAINT_FRAME_ROTATE_SAMPS)
    {
        mRotStepsCount = 0;
        mSwitchLockState = RELEASED;
        return;
    }

    Matrixd transMat;
    float intvlAngle = M_PI * 2 / mNumParas;
    float rotStepAngle = intvlAngle / ANIM_PARA_PAINT_FRAME_ROTATE_SAMPS;
    for (int i = 0; i < mNumParas; i++)
    {
        float phi = i * intvlAngle;
        if (mSwitchLockState == ROTATE_FORWARD)
            phi += -intvlAngle * (mStateParaIdx - 1) - rotStepAngle * mRotStepsCount;
        else if (mSwitchLockState == ROTATE_BACKWARD)
            phi += -intvlAngle * (mStateParaIdx + 1) + rotStepAngle * mRotStepsCount;

        Vec3 transVec = Vec3(0, -cos(phi), sin(phi)) * mSwitchRadius;
        transMat.makeTranslate(transVec);
        mParaEntryArray[i]->mMatrixTrans->setMatrix(transMat);
    }
}

