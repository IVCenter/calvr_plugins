/***************************************************************
* File Name: DesignObjectHandler.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Sep 10, 2010
*
***************************************************************/
#include "DesignObjectHandler.h"

using namespace std;
using namespace osg;


//Constructor
DesignObjectHandler::DesignObjectHandler(Group* rootGroup)
{
    // create object root: intersectable & non-intersectable
    mDesignObjectRoot = new Group();
    mNonIntersectableSceneGraphPtr = new Group();
    mIntersectableSceneGraphPtr = new Group();
    mRoot = rootGroup;

    mCAVEShapeSwitch = new Switch();
    mCAVEIconSurfaceSwitch = new Switch();
    mCAVEIconToolkitSwitch = new Switch();

    rootGroup->addChild(mDesignObjectRoot);
    mDesignObjectRoot->addChild(mNonIntersectableSceneGraphPtr);
    mDesignObjectRoot->addChild(mIntersectableSceneGraphPtr);

    mIntersectableSceneGraphPtr->addChild(mCAVEShapeSwitch);
    mIntersectableSceneGraphPtr->addChild(mCAVEIconSurfaceSwitch);
    mIntersectableSceneGraphPtr->addChild(mCAVEIconToolkitSwitch);

    mCAVEShapeSwitch->setAllChildrenOn();
    mCAVEIconSurfaceSwitch->setAllChildrenOn();
    mCAVEIconToolkitSwitch->setAllChildrenOn();

    // initialize design object pointer instances
    mDOGeometryCollector = new DOGeometryCollector();
    mDOGeometryCreator = new DOGeometryCreator();
    mDOGeometryEditor = new DOGeometryEditor();
    mDOGeometryCloner = new DOGeometryCloner();
    mDOGeometryEditor->setDOGeometryClonerPtr(mDOGeometryCloner);
    mDOGeometryEditor->setDOGeometryCollectorPtr(mDOGeometryCollector);
    mDOGeometryCloner->setDOGeometryCollectorPtr(mDOGeometryCollector);

    mVirtualScenicHandler = new VirtualScenicHandler(mNonIntersectableSceneGraphPtr, mIntersectableSceneGraphPtr);

    // all design objects geometry tools are sharing the same scene graph pointers from 'DesignObjectHandler'
    mDOGeometryCollector->initSceneGraphPtr(mNonIntersectableSceneGraphPtr, mIntersectableSceneGraphPtr,
			mCAVEShapeSwitch, mCAVEIconSurfaceSwitch, mCAVEIconToolkitSwitch);
    mDOGeometryCreator->initSceneGraphPtr(mNonIntersectableSceneGraphPtr, mIntersectableSceneGraphPtr,
			mCAVEShapeSwitch, mCAVEIconSurfaceSwitch, mCAVEIconToolkitSwitch);
    mDOGeometryEditor->initSceneGraphPtr(mNonIntersectableSceneGraphPtr, mIntersectableSceneGraphPtr,
			mCAVEShapeSwitch, mCAVEIconSurfaceSwitch, mCAVEIconToolkitSwitch);
    mDOGeometryCloner->initSceneGraphPtr(mNonIntersectableSceneGraphPtr, mIntersectableSceneGraphPtr,
			mCAVEShapeSwitch, mCAVEIconSurfaceSwitch, mCAVEIconToolkitSwitch);

    mDOGeometryCollector->initDesignObjects();
    mDOGeometryCreator->initDesignObjects();
    mDOGeometryEditor->initDesignObjects();
}

//Destructor
DesignObjectHandler::~DesignObjectHandler()
{
}


/***************************************************************
* Function: setActive()
*
* Description: Load activating/deactivating reference geometry
*
***************************************************************/
void DesignObjectHandler::setActive(bool flag)
{
    mVirtualScenicHandler->setGeometryVisible(flag);
/*    if (flag)
    {
        if (!mRoot->containsNode(mDesignObjectRoot) && mRoot && mDesignObjectRoot)
        {
            mRoot->addChild(mDesignObjectRoot);
        }
    }
    else
    {
        mRoot->removeChild(mDesignObjectRoot);
    }
    */
}


/***************************************************************
* Function: inputDevMoveEvent()
*
* Description: 
*
***************************************************************/
void DesignObjectHandler::inputDevMoveEvent()
{
}


/***************************************************************
* Function: inputDevPressEvent()
*
* Description: 
*
***************************************************************/
bool DesignObjectHandler::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
*
* Description: 
*
***************************************************************/
bool DesignObjectHandler::inputDevReleaseEvent()
{
    return false;
}


/***************************************************************
* Function: update
***************************************************************/
void DesignObjectHandler::update()
{
}

