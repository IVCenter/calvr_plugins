/***************************************************************
* File Name: DOGeometryCreator.cpp
*
* Description:
*
* Written by ZHANG Lelin on Jan 20, 2011
*
***************************************************************/
#include "DOGeometryCreator.h"


using namespace std;
using namespace osg;


// Constructor
DOGeometryCreator::DOGeometryCreator()
{
}


/***************************************************************
* Function: initDesignObjects()
***************************************************************/
void DOGeometryCreator::initDesignObjects()
{
    CAVEAnimationModeler::ANIMLoadGeometryCreatorReference(&mSnapWireframeSwitch, &mSnapSolidshapeSwitch);

    mCAVEGroupRefPlane = new CAVEGroupReferencePlane;
    mCAVEGroupRefAxis = new CAVEGroupReferenceAxis;

    /* Types of CAVEGeode derivatives that attached to 'mNonIntersectableSceneGraphPtr':
      'CAVEGeodeSnapWireframe', 'CAVEGeodeSnapSolidshape', 'CAVEGeodeReference', the
       only CAVEGeode derivative that attached to 'mIntersectableSceneGraphPtr' is the
       type 'CAVEGeodeShape' in DOGeometryCreator::registerSolidShape().
    */
    mNonIntersectableSceneGraphPtr->addChild(mSnapWireframeSwitch);
    mNonIntersectableSceneGraphPtr->addChild(mSnapSolidshapeSwitch);
    mNonIntersectableSceneGraphPtr->addChild(mCAVEGroupRefPlane);
    mNonIntersectableSceneGraphPtr->addChild(mCAVEGroupRefAxis);

    mWireframeActiveID = -1;		mWireframeGeode = NULL;
    mSolidShapeActiveID = -1;		mSolidshapeGeode = NULL;
}


/***************************************************************
* Function: setWireframeActiveID()
***************************************************************/
void DOGeometryCreator::setWireframeActiveID(const int &idx)
{
    mWireframeActiveID = idx;
    if (idx >= 0)
    {
        mSnapWireframeSwitch->setSingleChildOn(mWireframeActiveID);
        mWireframeGeode = dynamic_cast <CAVEGeodeSnapWireframe*> (mSnapWireframeSwitch->getChild(mWireframeActiveID));
    }
    else 
    {
        mSnapWireframeSwitch->setAllChildrenOff();
        mWireframeGeode = NULL;
    }
}


/***************************************************************
* Function: setSolidshapeActiveID()
***************************************************************/
void DOGeometryCreator::setSolidshapeActiveID(const int &idx)
{
    mSolidShapeActiveID = idx;
    if (idx >= 0)
    {
        mSnapSolidshapeSwitch->setSingleChildOn(mSolidShapeActiveID);
        mSolidshapeGeode = 
            dynamic_cast <CAVEGeodeSnapSolidshape*> (mSnapSolidshapeSwitch->getChild(mSolidShapeActiveID));
    }
    else 
    {
        mSnapSolidshapeSwitch->setAllChildrenOff();
        mSolidshapeGeode = NULL;
    }
}


/***************************************************************
* Function: setWireframeInitPos()
***************************************************************/
void DOGeometryCreator::setWireframeInitPos(const osg::Vec3 &initPos)
{
    if (mWireframeGeode) 
        mWireframeGeode->setInitPosition(initPos);
}


/***************************************************************
* Function: setSolidshapeInitPos()
***************************************************************/
void DOGeometryCreator::setSolidshapeInitPos(const osg::Vec3 &initPos, bool snap)
{
    if (mSolidshapeGeode) 
        mSolidshapeGeode->setInitPosition(initPos, snap);
}


/***************************************************************
* Function: resetWireframeGeodes()
*
* Set wireframe to central position of current design state 
* sphere with normal sizes, will call the following members:
*
* CAVEGeodeSnapWireframe::setInitPosition()
* CAVEGeodeSnapWireframe::setScaleVect()
* CAVEGeodeSnapWireframe::resize()
*
***************************************************************/
void DOGeometryCreator::resetWireframeGeodes(const osg::Vec3 &centerPos)
{
    if (mWireframeGeode)
    {
        Vec3 wireframeGridVect;
        float zoomfact = 1.5f;
        float unit = CAVEGeodeSnapWireframe::gSnappingUnitDist;
        if (mWireframeActiveID == CAVEGeodeShape::BOX)
        {
            float len = ANIM_VIRTUAL_SPHERE_RADIUS / 0.9 * zoomfact;
            len = ((int) (len / unit)) * unit;
            Vec3 scaleBox = Vec3(len, len, len);
            mWireframeGeode->setInitPosition(centerPos + Vec3(-len, -len, -len) * 0.5);
            mWireframeGeode->setScaleVect(scaleBox);
        }
        else if (mWireframeActiveID == CAVEGeodeShape::CYLINDER)
        {
            float rad = ANIM_VIRTUAL_SPHERE_RADIUS / 1.5 * zoomfact;
            rad = ((int) (rad / unit)) * unit;
            Vec3 scaleCylinder = Vec3(rad, 0, rad * 2);
            mWireframeGeode->setInitPosition(centerPos + Vec3(0, 0, -rad));
            mWireframeGeode->setScaleVect(scaleCylinder);
        }
        mWireframeGeode->resize(wireframeGridVect);
    }
}


/***************************************************************
* Function: setReferencePlaneMasking()
***************************************************************/
void DOGeometryCreator::setReferencePlaneMasking(bool flagXY, bool flagXZ, bool flagYZ)
{
    mCAVEGroupRefPlane->setPlaneMasking(flagXY, flagXZ, flagYZ);
}


/***************************************************************
* Function: setReferenceAxisMasking()
***************************************************************/
void DOGeometryCreator::setReferenceAxisMasking(bool flag)
{
    mCAVEGroupRefAxis->setAxisMasking(flag);
}


/***************************************************************
* Function: setScalePerUnit()
*
* Function called by 'DSGeometryCreator::switchToPrevSubState()'
* and 'DSGeometryCreator::switchToNextSubState()' when drawing
* state is not 'IDLE'. Texts on reference axis will be updated
* accordingly if a shape is being created.
*
***************************************************************/
void DOGeometryCreator::setScalePerUnit(const float &scalePerUnit, const string &infoStr)
{
    if (mSolidshapeGeode) 
        mSolidshapeGeode->setSnappingUnitDist(scalePerUnit);

    setResize();

    mCAVEGroupRefPlane->setUnitGridSize(scalePerUnit);
    mCAVEGroupRefAxis->updateUnitGridSizeInfo(infoStr);

    updateReferenceAxis();
}


/***************************************************************
* Function: updateReferenceAxis()
*
* Update axis position and orientation based on initial position
* center of 'mWireframeGeode' and size of 'mSolidshapeGeode'
*
***************************************************************/
void DOGeometryCreator::updateReferenceAxis()
{
    if (mCAVEGroupRefAxis->isVisible())
    {
        if (mWireframeGeode)
        {
            mCAVEGroupRefAxis->setCenterPos(mWireframeGeode->getInitPosition());
            if (mSolidshapeGeode)
            {
                mCAVEGroupRefAxis->updateDiagonal(mWireframeGeode->getDiagonalVect(), 
                              mSolidshapeGeode->getScaleVect());
            }
        }
    }
}


/***************************************************************
* Function: updateReferencePlane()
*
* Round intersected vector 'center' to closest snapping point
*
***************************************************************/
void DOGeometryCreator::updateReferencePlane(const osg::Vec3 &center, bool noSnap)
{
    mCAVEGroupRefPlane->setPlaneMasking(true, true, true);
    mCAVEGroupRefPlane->setCenterPos(center, noSnap);
}


/***************************************************************
* Function: setPointerDir()
***************************************************************/
void DOGeometryCreator::setPointerDir(const osg::Vec3 &pointerDir)
{
    CAVEGroupReference::setPointerDir(pointerDir);
}


/***************************************************************
* Function: setSnapPos()
***************************************************************/
void DOGeometryCreator::setSnapPos(const osg::Vec3 &snapPos, bool snap)
{
    if (mWireframeGeode) 
        mWireframeGeode->setScaleVect(snapPos - mWireframeGeode->getInitPosition());
    setResize(snap);
}


/***************************************************************
* Function: setResize()
***************************************************************/
void DOGeometryCreator::setResize(const float &s)
{
    if (mWireframeGeode) 
        mWireframeGeode->setScaleVect(Vec3(s, s, s));
    setResize();
}


/***************************************************************
* Function: setResize()
***************************************************************/
void DOGeometryCreator::setResize(bool snap)
{
    Vec3 wireframeGridVect = Vec3(0, 0, 0);
    if (mWireframeGeode) 
        mWireframeGeode->resize(wireframeGridVect);

    if (mSolidshapeGeode) 
    {
        mSolidshapeGeode->resize(wireframeGridVect, snap);
    }
}


/***************************************************************
* Function: registerSolidShape()
***************************************************************/
void DOGeometryCreator::registerSolidShape()
{
    if (mSolidshapeGeode->isValid())
    {
        // 'mDOShapeSwitch -> CAVEGroupShape -> CAVEGeodeShape' each CAVEGroupShape contains
        //  only one instance of 'CAVEGeodeShape' at the time of being created.
        CAVEGeodeShape *shape = new CAVEGeodeShape((CAVEGeodeShape::Type) mSolidShapeActiveID, 
                mSolidshapeGeode->getInitPosition(), mSolidshapeGeode->getScaleVect());
        CAVEGroupShape *group = new CAVEGroupShape(shape);
        mDOShapeSwitch->addChild(group);
    }
}

