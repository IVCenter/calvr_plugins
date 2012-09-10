/***************************************************************
* File Name: DSVirtualSphere.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Oct 5, 2010
*
***************************************************************/
#include "DSVirtualSphere.h"

using namespace std;
using namespace osg;

//Constructor
DSVirtualSphere::DSVirtualSphere()
{
    CAVEAnimationModeler::ANIMCreateVirtualSphere(&mPATransFwd, &mPATransBwd);
    this->addChild(mPATransFwd);	//  child #0
    this->addChild(mPATransBwd);	//  child #1

    fwdVec.push_back(mPATransFwd);
    bwdVec.push_back(mPATransBwd);

    setSingleChildOn(0);

    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    mIsOpen = false;
}


void DSVirtualSphere::addChildState(DesignStateBase* ds)
{
    if (!ds)
        return;

    mChildStates.push_back(ds);

    float z = mChildStates.size();
    z = -z + 0.5;
    
    osg::Vec3 startPos(-1.5, 0, 0);
    osg::Vec3 pos(-1.5, 0, z);
    AnimationPath* animationPathScaleFwd = new AnimationPath();
    AnimationPath* animationPathScaleBwd = new AnimationPath();
    animationPathScaleFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationPathScaleBwd->setLoopMode(AnimationPath::NO_LOOPING);

    Vec3 scaleFwd, scaleBwd;
    float step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
    for (int j = 0; j < ANIM_VIRTUAL_SPHERE_NUM_SAMPS + 1; j++)
    {
        float val = j * step;
        scaleFwd = Vec3(val, val, val);
        scaleBwd = Vec3(1-val, 1-val, 1-val);

        osg::Vec3 diff = startPos - pos;
        osg::Vec3 fwd, bwd;

        for (int i = 0; i < 3; ++i)
            diff[i] *= val;

        fwd = startPos - diff;
        bwd = pos + diff;

        animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(fwd, Quat(), scaleFwd));
        animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(bwd, Quat(), scaleBwd));
    }

    AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
                        0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
                        0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 

    osg::PositionAttitudeTransform *fwd, *bwd;
    fwd = new osg::PositionAttitudeTransform();
    bwd = new osg::PositionAttitudeTransform();

    fwd->setUpdateCallback(animCallbackFwd);
    bwd->setUpdateCallback(animCallbackBwd);

    fwd->addChild(ds);
    bwd->addChild(ds);

    fwdVec.push_back(fwd);
    bwdVec.push_back(bwd);

    this->addChild(fwd);
    this->addChild(bwd);

    setAllChildrenOn();
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSVirtualSphere::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    if (flag) 
    {
        setAllChildrenOn();
    }
    else 
    {
    }

    if (!mPATransFwd || !mPATransBwd) 
        return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
        setSingleChildOn(0);
        animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mPATransFwd);
        //mDOIntersector->loadRootTargetNode(gDesignObjectRootGroup, NULL);
    } 
    else 
    {
        //setSingleChildOn(1);
        //animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());
    }

    if (animCallback) 
        animCallback->reset();
}


/***************************************************************
* Function: switchToPrevState()
***************************************************************/
void DSVirtualSphere::switchToPrevSubState()
{
    AnimationPathCallback* animCallback = NULL;
    setAllChildrenOff();
    if (mIsOpen)
    {
        setSingleChildOn(0);
        for (int i = 1; i < bwdVec.size(); ++i)
        {
            setChildValue(bwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (bwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, bwdVec[i]);

            if (animCallback)
                animCallback->reset();
        }

        std::list<DesignStateBase*>::iterator it;
        for (it = mChildStates.begin(); it != mChildStates.end(); ++it)
        {
            (*it)->setObjectEnabled(true);
        }
        mIsOpen = false;
    }
    else
    {
        setSingleChildOn(0);
        for (int i = 1; i < fwdVec.size(); ++i)
        {
            setChildValue(fwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (fwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[i]);

            if (animCallback)
                animCallback->reset();
        }
        std::list<DesignStateBase*>::iterator it;
        for (it = mChildStates.begin(); it != mChildStates.end(); ++it)
        {
            (*it)->setObjectEnabled(true);
        }

        mIsOpen = true;
    }
}


/***************************************************************
* Function: switchToNextState()
***************************************************************/
void DSVirtualSphere::switchToNextSubState()
{
    switchToPrevSubState();
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSVirtualSphere::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;
    std::list<DesignStateBase*>::iterator it;
    it = mChildStates.begin();

    for (int i = 0; i < fwdVec.size(); ++i)
    {
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[i]->getChild(0));
        //setChildValue(fwdVec[i], true);

        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            // open/close menu
            if (i == 0)
            {
                switchToPrevSubState();
            }
            // pass on event
            else
            {
                (*it)->inputDevPressEvent(pointerOrg, pointerPos);
            }
        }
        it++;
    }
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[0]->getChild(0));
    return false;
}

