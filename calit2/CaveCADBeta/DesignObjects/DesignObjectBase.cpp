/***************************************************************
* File Name: DesignObjectBase.cpp
*
* Description: Implementation of base design object classes
*
* Written by ZHANG Lelin on Jan 19, 2011
*
***************************************************************/
#include "DesignObjectBase.h"


using namespace std;
using namespace osg;


// Constructor
DesignObjectBase::DesignObjectBase()
{
}


/***************************************************************
* Function: initSceneGraphPtr()
***************************************************************/
void DesignObjectBase::initSceneGraphPtr(Group *nonInterSCPtr, Group *interSCPtr,
	Switch *shapeSwitch, Switch *surfaceIconSwitch, Switch *toolkitIconSwitch)
{
    mNonIntersectableSceneGraphPtr = nonInterSCPtr;
    mIntersectableSceneGraphPtr = interSCPtr;

    mDOShapeSwitch = shapeSwitch;
    mDOIconSurfaceSwitch = surfaceIconSwitch;
    mDOIconToolkitSwitch = toolkitIconSwitch;
}






