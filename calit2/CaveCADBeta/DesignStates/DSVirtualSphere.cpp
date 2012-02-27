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

    setAllChildrenOff();
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
    mDSParticleSystemPtr->setEmitterEnabled(flag);
    if (flag) setAllChildrenOn();
    else setAllChildrenOff();

    if (!mPATransFwd || !mPATransBwd) return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
	setSingleChildOn(0);
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
    } else {
	setSingleChildOn(1);
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());
    }
    if (animCallback) animCallback->reset();
}


/***************************************************************
* Function: switchToPrevState()
***************************************************************/
void DSVirtualSphere::switchToPrevSubState()
{
}


/***************************************************************
* Function: switchToNextState()
***************************************************************/
void DSVirtualSphere::switchToNextSubState()
{
}






















