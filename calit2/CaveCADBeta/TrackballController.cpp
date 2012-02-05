/***************************************************************
* File Name: TrackballController.cpp
*
* Description: Implementation of single axis trackball control
*
* Written by ZHANG Lelin on Oct 20, 2010
*
***************************************************************/
#include "TrackballController.h"


using namespace osg;
using namespace std;

// Constructor
TrackballController::TrackballController(): mActiveFlag(false), mInitialPickFlag(false), mAxis(Vec3(0, 0, 1))
{
    mRefPoint = Vec3(0, -1, 0);
    mCtrPoint = Vec3(0, -1, 0);
}


/***************************************************************
* Function: triggerInitialPick()
***************************************************************/
void TrackballController::triggerInitialPick()
{
    if (mActiveFlag) mInitialPickFlag = true; 
    else mInitialPickFlag = false; 
}


/***************************************************************
* Function: updateCtrPoint()
***************************************************************/
void TrackballController::updateCtrPoint(const Vec3 &ctrPoint)
{
    if (!mActiveFlag) return;

    if (mInitialPickFlag)
    {
	mRefPoint = mCtrPoint = ctrPoint;
	mInitialPickFlag = false; 
    }
    else
    {
	mRefPoint = mCtrPoint;
	mCtrPoint = ctrPoint;	
    }
}


/***************************************************************
* Function: getAngularOffset()
*
* Description: Returns the angular value in radius indicating
* how much the trackball is rotated around its axis from
* reference point to current control point
*
***************************************************************/
float TrackballController::getAngularOffset()
{
    if (!mActiveFlag) return 0.f;

    Vec3f refProj = mRefPoint - mAxis * (mRefPoint * mAxis);
    Vec3f ctrProj = mCtrPoint - mAxis * (mCtrPoint * mAxis);
    if (refProj.length2() == 0 || ctrProj.length2() == 0) return 0.f;
    refProj.normalize();
    ctrProj.normalize();

    float dotproduct = refProj * ctrProj;
    if (dotproduct >= 1.0f || dotproduct <= -1.0f) return 0.f;

    float angle = acos(dotproduct);

    Vec3 rotDirAxis = refProj ^ ctrProj;
    if (mAxis * (refProj ^ ctrProj) < 0) angle = -angle;

    return angle;
}














