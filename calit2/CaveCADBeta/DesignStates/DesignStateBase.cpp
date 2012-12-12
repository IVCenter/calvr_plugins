/***************************************************************
* File Name: DesignStateBase.cpp
*
* Description: Implementation of base design state class
*
* Written by ZHANG Lelin on Oct 6, 2010
*
***************************************************************/
#include "DesignStateBase.h"

using namespace std;
using namespace osg;

Group *DesignStateBase::gDesignStateRootGroup(NULL);
Group *DesignStateBase::gDesignObjectRootGroup(NULL);

Vec3 DesignStateBase::gDesignStateCenterPos(Vec3(0, -0.5, 0));
Vec3 DesignStateBase::gDesignStateFrontVect(Vec3(0, 1, 0));
Matrixd DesignStateBase::gDesignStateBaseRotMat(Matrixd(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1));

// Constructor
DesignStateBase::DesignStateBase(): mObjEnabledFlag(false), mDevPressedFlag(false), mLockedFlag(false),
				    mUpperDSSwitchFuncPtr(NULL), mLowerDSSwitchFuncPtr(NULL)
{
    mDSParticleSystemPtr = NULL;
    mPATransFwd = NULL;
    mPATransBwd = NULL;

    mDSIntersector = NULL;
    mDOIntersector = NULL;

    mUpperDSVector.clear();
    mLowerDSVector.clear();
}


/***************************************************************
* Function: switchToUpperDesignState()
***************************************************************/
void DesignStateBase::switchToUpperDesignState(const int &idx)
{
    if (mUpperDSVector.size() > idx) mUpperDSSwitchFuncPtr(idx);
}


/***************************************************************
* Function: switchToLowerDesignState()
***************************************************************/
void DesignStateBase::switchToLowerDesignState(const int &idx)
{
    if (mLowerDSVector.size() > idx) mLowerDSSwitchFuncPtr(idx);
}


/***************************************************************
* Function: setDesignStateRootGroupPtr()
***************************************************************/
void DesignStateBase::setDesignStateRootGroupPtr(osg::Group *designStateRootGroup)
{
    gDesignStateRootGroup = designStateRootGroup;
}


/***************************************************************
* Function: setDesignObjectRootGroupPtr()
***************************************************************/
void DesignStateBase::setDesignObjectRootGroupPtr(osg::Group *designObjectRootGroup)
{
    gDesignObjectRootGroup = designObjectRootGroup;
}


/***************************************************************
* Function: setDesignStateCenterPos()
***************************************************************/
void DesignStateBase::setDesignStateCenterPos(const Vec3 &pos)
{
    gDesignStateCenterPos = pos;
}


/***************************************************************
* Function: setDesignStateFrontVect()
***************************************************************/
void DesignStateBase::setDesignStateFrontVect(const Vec3 &front)
{
    gDesignStateFrontVect = front;
    gDesignStateBaseRotMat.makeRotate(Vec3(0, 1, 0), gDesignStateFrontVect);
}


/***************************************************************
* Function: test()
***************************************************************/
bool DesignStateBase::test(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    return mDSIntersector->test(pointerOrg, pointerPos);
}

