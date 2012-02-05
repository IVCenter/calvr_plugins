/***************************************************************
* File Name: CAVEGroupEditWireframe.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Mar 31, 2011
*
***************************************************************/
#include "CAVEGroupEditWireframe.h"


using namespace std;
using namespace osg;


// Constructor: CAVEGroupEditGeometryWireframe
CAVEGroupEditGeometryWireframe::CAVEGroupEditGeometryWireframe(): mRefShapeCenter(Vec3(0, 0, 0))
{
    mRootTrans = new MatrixTransform();
    mSwitch = new Switch();
    mSwitch->setAllChildrenOff();

    addChild(mRootTrans);
    mRootTrans->addChild(mSwitch);
    mMatTransVector.clear();

    mGeometryReferencePtr = NULL;
}


/***************************************************************
* Function: acceptCAVEGeometry()
*
* 'refShapeCenter' is decided when the first 'CAVEGeodeShape' is
* selected. Editting on geometry level won't change this value.
*
***************************************************************/
void CAVEGroupEditGeometryWireframe::acceptCAVEGeometry(CAVEGeometry *geometry, const osg::Vec3 &refShapeCenter)
{
    mRefShapeCenter = refShapeCenter;
    mGeometryReferencePtr = geometry;

    /* set values of 'mBoudingRadius', 'mBoundBoxScaleMat', 'mBoundSphereScaleMat' for text display */
    const BoundingBox& bb = geometry->getBound();

    float xmin, ymin, zmin, xmax, ymax, zmax, radius;
    xmin = bb.xMin();		xmax = bb.xMax();
    ymin = bb.yMin();		ymax = bb.yMax();
    zmin = bb.zMin();		zmax = bb.zMax();
    radius = bb.radius();

    mBoundingRadius = radius;
    mBoundBoxScaleMat.makeScale(Vec3(xmax-xmin, ymax-ymin, zmax-zmin));
    mBoundSphereScaleMat.makeScale(Vec3(radius, radius, radius));

    /* apply 'refShapeCenter' offset at root level */
    Matrixd offsetMat;
    offsetMat.makeTranslate(-refShapeCenter);
    mRootTrans->setMatrix(offsetMat);

    /* create wireframe geode[0] from vertex list of reference 'geometry' */
    MatrixTransform *editTrans = new MatrixTransform;
    CAVEGeodeEditGeometryWireframe *wireframeGeode = new CAVEGeodeEditGeometryWireframe(geometry);
    mSwitch->addChild(editTrans);
    editTrans->addChild(wireframeGeode);
    mMatTransVector.push_back(editTrans);
}


/***************************************************************
* Function: updateCAVEGeometry()
*
* update 'CAVEGeometry' of the first 'GeometryWireframe' object
* in the group, other 'GeometryWireframe' are re-created during
* the editing process
*
***************************************************************/
void CAVEGroupEditGeometryWireframe::updateCAVEGeometry()
{
    CAVEGeodeEditGeometryWireframe *refGeometryWireframe =
	dynamic_cast <CAVEGeodeEditGeometryWireframe*> (mMatTransVector[0]->getChild(0));
    if (refGeometryWireframe)
    {
	refGeometryWireframe->getCAVEGeometryPtr()->dirtyDisplayList();
	refGeometryWireframe->getCAVEGeometryPtr()->dirtyBound();
    }
}


/***************************************************************
* Function: applyTranslation()
*
* 'gridSVect': number of snapping segments along each direction
* 'gridUnitLegnth': actual length represented by each segment,
*  only one component of 'gridSVect' is supposed to be non-zero
*
***************************************************************/
void CAVEGroupEditGeometryWireframe::applyTranslation(const osg::Vec3s &gridSVect, const float &gridUnitLegnth,
							const string &gridUnitLegnthInfo)
{
    mSwitch->setAllChildrenOn();

    /* move to other direction: clear all children of 'mSwitch' except child[0], rebuild offset tree */
    if ((mMoveSVect.x() * gridSVect.x() + mMoveSVect.y() * gridSVect.y() + mMoveSVect.z() * gridSVect.z()) <= 0)
    {
	unsigned int numChildren = mSwitch->getNumChildren();
	if (numChildren > 1)
	{
	    mSwitch->removeChildren(1, numChildren - 1);
	    MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	    mMatTransVector.erase(itrMatTrans + 1, itrMatTrans + numChildren);
	}
	mMoveSVect = Vec3s(0, 0, 0);
    }

    /* decide unit offset vector and start/end index of children under 'mSwitch' */
    Vec3 gridOffsetVect;
    short idxStart = 0, idxEnd = 0;
    if (gridSVect.x() != 0)
    {
	idxStart = mMoveSVect.x();
	idxEnd = gridSVect.x();
	if (gridSVect.x() > 0) gridOffsetVect = Vec3(gridUnitLegnth, 0, 0);
	else gridOffsetVect = Vec3(-gridUnitLegnth, 0, 0);
    }
    else if (gridSVect.y() != 0)
    {
	idxStart = mMoveSVect.y();
	idxEnd = gridSVect.y();
	if (gridSVect.y() > 0) gridOffsetVect = Vec3(0, gridUnitLegnth, 0);
	else gridOffsetVect = Vec3(0, -gridUnitLegnth, 0);
    }
    else if (gridSVect.z() != 0)
    {
	idxStart = mMoveSVect.z();
	idxEnd = gridSVect.z();
	if (gridSVect.z() > 0) gridOffsetVect = Vec3(0, 0, gridUnitLegnth);
	else gridOffsetVect = Vec3(0, 0, -gridUnitLegnth);
    } 
    idxStart = idxStart > 0 ? idxStart: -idxStart;
    idxEnd = idxEnd > 0 ? idxEnd : -idxEnd;

    /* create or remove a sequence of extra children under 'mSwitch' */
    if (idxStart < idxEnd)
    {
	for (short i = idxStart + 1; i <= idxEnd; i++)
	{
	    Matrixd offsetMat;
	    offsetMat.makeTranslate(gridOffsetVect * i);
	    MatrixTransform *offsetTrans = new MatrixTransform;
	    CAVEGeodeEditGeometryWireframe *wireframeGeode = new CAVEGeodeEditGeometryWireframe(mGeometryReferencePtr);
	    mSwitch->addChild(offsetTrans);
	    mMatTransVector.push_back(offsetTrans);
	    offsetTrans->addChild(wireframeGeode);
	    offsetTrans->setMatrix(offsetMat);
	}
    }
    else if (idxStart > idxEnd)
    {
	mSwitch->removeChildren(idxEnd + 1, idxStart - idxEnd);
	MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	itrMatTrans += idxEnd + 1;
	mMatTransVector.erase(itrMatTrans, itrMatTrans + (idxStart - idxEnd));
    }

    mMoveSVect = gridSVect;

    if (!mPrimaryFlag) return;

    /* update info text if 'this' wireframe is primary */
    mEditInfoTextSwitch->setAllChildrenOn();

    char info[128];
    sprintf(info, "Offset = %3.2f m\nSnapping = ", gridUnitLegnth * idxEnd);
    mEditInfoText->setText(info + gridUnitLegnthInfo);
}


/***************************************************************
* Function: applyRotation()
*
* 'axisSVect': rotational snapping values around each axis,
*  for instance, Vec3(2, 0, 0) means rotation around X-axis by
*  two times of 'gridUnitAngle'
*  only one component of 'axisIntVect' is supposed to be non-zero
*
***************************************************************/
void CAVEGroupEditGeometryWireframe::applyRotation(const osg::Vec3s &axisSVect, const float &gridUnitAngle,
						const string &gridUnitAngleInfo)
{
    mSwitch->setAllChildrenOn();

    /* rotate in other direction: clear all children of 'mSwitch' except child[0], rebuild offset tree */
    if ((mRotateSVect.x() * axisSVect.x() + mRotateSVect.y() * axisSVect.y() + mRotateSVect.z() * axisSVect.z()) <= 0)
    {
	unsigned int numChildren = mSwitch->getNumChildren();
	if (numChildren > 1)
	{
	    mSwitch->removeChildren(1, numChildren - 1);
	    MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	    mMatTransVector.erase(itrMatTrans + 1, itrMatTrans + numChildren);
	}
	mRotateSVect = Vec3s(0, 0, 0);
    }

    /* decide unit rotation quat and start/end index of children under 'mSwitch' */
    Vec3 gridRotationAxis;
    short idxStart = 0, idxEnd = 0;
    if (axisSVect.x() != 0)
    {
	idxStart = mRotateSVect.x();
	idxEnd = axisSVect.x();
	if (axisSVect.x() > 0) gridRotationAxis = Vec3(1, 0, 0);
	else gridRotationAxis = Vec3(-1, 0, 0);
    }
    else if (axisSVect.y() != 0)
    {
	idxStart = mRotateSVect.y();
	idxEnd = axisSVect.y();
	if (axisSVect.y() > 0) gridRotationAxis = Vec3(0, 1, 0);
	else gridRotationAxis = Vec3(0, -1, 0);
    }
    else if (axisSVect.z() != 0)
    {
	idxStart = mRotateSVect.z();
	idxEnd = axisSVect.z();
	if (axisSVect.z() > 0) gridRotationAxis = Vec3(0, 0, 1);
	else gridRotationAxis = Vec3(0, 0, -1);
    }
    idxStart = idxStart > 0 ? idxStart: -idxStart;
    idxEnd = idxEnd > 0 ? idxEnd : -idxEnd;

    /* create or remove a sequence of extra children under 'mSwitch' */
    if (idxStart < idxEnd)
    {
	for (short i = idxStart + 1; i <= idxEnd; i++)
	{
	    Matrixd rotMat, offsetMat, offsetMatRev;
	    rotMat.makeRotate(gridUnitAngle * i, gridRotationAxis);
	    offsetMatRev.makeTranslate(-mRefShapeCenter);
	    offsetMat.makeTranslate(mRefShapeCenter);
	    MatrixTransform *rotateTrans = new MatrixTransform;
	    CAVEGeodeEditGeometryWireframe *wireframeGeode = new CAVEGeodeEditGeometryWireframe(mGeometryReferencePtr);
	    mSwitch->addChild(rotateTrans);
	    mMatTransVector.push_back(rotateTrans);
	    rotateTrans->addChild(wireframeGeode);
	    rotateTrans->setMatrix(offsetMatRev * rotMat * offsetMat);
	}
    }
    else if (idxStart > idxEnd)
    {
	mSwitch->removeChildren(idxEnd + 1, idxStart - idxEnd);
	MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	itrMatTrans += idxEnd + 1;
	mMatTransVector.erase(itrMatTrans, itrMatTrans + (idxStart - idxEnd));
    }

    mRotateSVect = axisSVect;

    if (!mPrimaryFlag) return;

    /* update info text if 'this' wireframe is primary */
    mEditInfoTextSwitch->setAllChildrenOn();

    char info[128];
    const float gridUnitAngleDegree = gridUnitAngle * 180 / M_PI;
    sprintf(info, "Angle = %3.2f\nSnapping = ", gridUnitAngleDegree * idxEnd);
    mEditInfoText->setText(info + gridUnitAngleInfo);
}


/***************************************************************
* Function: applyScaling()
***************************************************************/
void CAVEGroupEditGeometryWireframe::applyScaling(const short &nOffsetSegs, const Vec3 &gridUnitScaleVect, 
						const std::string &gridUnitScaleInfo)
{
    mSwitch->setAllChildrenOn();

    /* scale to other direction: clear all children of 'mSwitch' except child[0], rebuild offset tree */
    if (mScaleNumSegs * nOffsetSegs <= 0)
    {
	unsigned int numChildren = mSwitch->getNumChildren();
	if (numChildren > 1)
	{
	    mSwitch->removeChildren(1, numChildren - 1);
	    MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	    mMatTransVector.erase(itrMatTrans + 1, itrMatTrans + numChildren);
	}
	mScaleNumSegs = 0;
    }

    /* decide unit offset vector and start/end index of children under 'mManipulateSwitch' */
    short idxStart = 0, idxEnd = 0;
    if (nOffsetSegs != 0)
    {
	idxStart = mScaleNumSegs;
	idxEnd = nOffsetSegs;
    }
    idxStart = idxStart > 0 ? idxStart: -idxStart;
    idxEnd = idxEnd > 0 ? idxEnd : -idxEnd;

    /* create or remove a sequence of extra children under 'mSwitch' */
    if (idxStart < idxEnd)
    {
	for (short i = idxStart + 1; i <= idxEnd; i++)
	{
	    /* generate scaling vector with non-negative values */
	    Vec3 scaleVect = Vec3(1, 1, 1);
	    if (nOffsetSegs > 0) scaleVect += gridUnitScaleVect * i;
	    else scaleVect -= gridUnitScaleVect * i;
	    scaleVect.x() = scaleVect.x() > 0 ? scaleVect.x() : 0;
	    scaleVect.y() = scaleVect.y() > 0 ? scaleVect.y() : 0;
	    scaleVect.z() = scaleVect.z() > 0 ? scaleVect.z() : 0;

	    Matrixd scaleMat, offsetMat, offsetMatRev;
	    scaleMat.makeScale(scaleVect);
	    offsetMatRev.makeTranslate(-mRefShapeCenter);
	    offsetMat.makeTranslate(mRefShapeCenter);

	    CAVEGeodeEditGeometryWireframe *wireframeGeode = new CAVEGeodeEditGeometryWireframe(mGeometryReferencePtr);
	    MatrixTransform *offsetTrans = new MatrixTransform;
	    mSwitch->addChild(offsetTrans);
	    offsetTrans->addChild(wireframeGeode);
	    offsetTrans->setMatrix(offsetMatRev * scaleMat * offsetMat);
	    mMatTransVector.push_back(offsetTrans);
	}
    }
    else if (idxStart > idxEnd)
    {
	mSwitch->removeChildren(idxEnd + 1, idxStart - idxEnd);
	MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	itrMatTrans += idxEnd + 1;
	mMatTransVector.erase(itrMatTrans, itrMatTrans + (idxStart - idxEnd));
    }

    mScaleNumSegs = nOffsetSegs;
    mScaleUnitVect = gridUnitScaleVect;

    if (!mPrimaryFlag) return;

    /* update info text if 'this' wireframe is primary */
    mEditInfoTextSwitch->setAllChildrenOn();

    float scaleUnit = gridUnitScaleVect.x() > 0 ? gridUnitScaleVect.x() : gridUnitScaleVect.y();
    scaleUnit = scaleUnit > 0 ? scaleUnit : gridUnitScaleVect.z();

    char info[128];
    sprintf(info, "Scale = %3.2f \nSnapping = ", scaleUnit * idxEnd + 1.0f);
    mEditInfoText->setText(info + gridUnitScaleInfo);
}


/***************************************************************
* Function: updateGridUnits()
***************************************************************/
void CAVEGroupEditGeometryWireframe::updateGridUnits(const float &gridUnitLegnth, 
			const float &gridUnitAngle, const float &gridUnitScale)
{
    const unsigned int numChildren = mSwitch->getNumChildren();
    if (numChildren <= 1) return;

    /* use 'mRotateSVect', 'mMoveSVect' to discriminate between editting states */
    if (mMoveSVect.x() != 0 || mMoveSVect.y() != 0 || mMoveSVect.z() != 0)
    {
	Vec3 gridOffsetVect = Vec3(0, 0, 0);
	if (mMoveSVect.x() > 0) gridOffsetVect = Vec3(gridUnitLegnth, 0, 0);
	else if (mMoveSVect.x() < 0) gridOffsetVect = Vec3(-gridUnitLegnth, 0, 0);
	else if (mMoveSVect.y() > 0) gridOffsetVect = Vec3(0,  gridUnitLegnth, 0);
	else if (mMoveSVect.y() < 0) gridOffsetVect = Vec3(0, -gridUnitLegnth, 0);
	else if (mMoveSVect.z() > 0) gridOffsetVect = Vec3(0, 0,  gridUnitLegnth);
	else if (mMoveSVect.z() < 0) gridOffsetVect = Vec3(0, 0, -gridUnitLegnth);

	for (int i = 1; i < numChildren; i++)
	{
	    Matrixd offsetMat;
	    offsetMat.makeTranslate(gridOffsetVect * i);
	    mMatTransVector[i]->setMatrix(offsetMat);
	}
    }
    else if (mRotateSVect.x() != 0 || mRotateSVect.y() != 0 || mRotateSVect.z() != 0)
    {
	Vec3 gridRotationAxis = Vec3(0, 0, 1);
	if (mRotateSVect.x() > 0) gridRotationAxis = Vec3(1, 0, 0);
	else if (mRotateSVect.x() < 0) gridRotationAxis = Vec3(-1, 0, 0);
	else if (mRotateSVect.y() > 0) gridRotationAxis = Vec3(0, 1, 0);
	else if (mRotateSVect.y() < 0) gridRotationAxis = Vec3(0, -1, 0);
	else if (mRotateSVect.z() > 0) gridRotationAxis = Vec3(0, 0, 1);
	else if (mRotateSVect.z() < 0) gridRotationAxis = Vec3(0, 0, -1);

	for (int i = 0; i < numChildren; i++)
	{
	    Matrixd rotMat, offsetMat, offsetMatRev;
	    rotMat.makeRotate(gridUnitAngle * i, gridRotationAxis);
	    offsetMatRev.makeTranslate(-mRefShapeCenter);
	    offsetMat.makeTranslate(mRefShapeCenter);
	    mMatTransVector[i]->setMatrix(offsetMatRev * rotMat * offsetMat);
	}
    }
    else if (mScaleNumSegs != 0)
    {
	/* rewrite scaling vector */
	mScaleUnitVect.x() = mScaleUnitVect.x() > 0 ? 1:0;
	mScaleUnitVect.y() = mScaleUnitVect.y() > 0 ? 1:0;
	mScaleUnitVect.z() = mScaleUnitVect.z() > 0 ? 1:0;
	mScaleUnitVect *= gridUnitScale;

	for (int i = 0; i < numChildren; i++)
	{
	    /* generate scaling vector with non-negative values */
	    Vec3 scaleVect = Vec3(1, 1, 1);
	    if (mScaleNumSegs > 0) scaleVect += mScaleUnitVect * i;
	    else scaleVect -= mScaleUnitVect * i;
	    scaleVect.x() = scaleVect.x() > 0 ? scaleVect.x() : 0;
	    scaleVect.y() = scaleVect.y() > 0 ? scaleVect.y() : 0;
	    scaleVect.z() = scaleVect.z() > 0 ? scaleVect.z() : 0;

	    Matrixd scaleMat;
	    scaleMat.makeScale(scaleVect);
	    mMatTransVector[i]->setMatrix(scaleMat);
	}
    }
}


/***************************************************************
* Function: applyEditorInfo()
*
* Surface level editting does not change global properties of
* 'CAVEGroupEditGeometryWireframe', leave the func body empty.
*
***************************************************************/
void CAVEGroupEditGeometryWireframe::applyEditorInfo(CAVEGeodeShape::EditorInfo **infoPtr)
{
}


/***************************************************************
* Function: clearActiveWireframes()
***************************************************************/
void CAVEGroupEditGeometryWireframe::clearActiveWireframes()
{
    /* clear all wireframe geodes except the very first one in both 'mSwitch' and 'mMatTransVector' */
    unsigned int numActiveChildren = mSwitch->getNumChildren();
    if (numActiveChildren > 1)
    {
	mSwitch->removeChildren(1, numActiveChildren - 1);
	MatrixTransVector::iterator itrMatTrans = mMatTransVector.begin();
	mMatTransVector.erase(itrMatTrans + 1, mMatTransVector.end());
    }

    /* clear surface level editting records */
    mMoveSVect = Vec3s(0, 0, 0);
    mRotateSVect = Vec3s(0, 0, 0);
    mScaleUnitVect = Vec3(0, 0, 0);
    mScaleNumSegs = 0;

    /* turn off all wireframes and info text */
    mSwitch->setAllChildrenOff();
    mEditInfoTextSwitch->setAllChildrenOff();
}









