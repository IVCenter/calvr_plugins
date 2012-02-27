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


// Constructor: CAVEGroupEditGeodeWireframe
CAVEGroupEditGeodeWireframe::CAVEGroupEditGeodeWireframe(): mRefShapeCenter(Vec3(0, 0, 0))
{
    /* create editting wireframe objects */
    mMoveSwitch = new Switch();
    mRotateSwitch = new Switch();
    mManipulateSwitch = new Switch();
    mMoveSwitch->setAllChildrenOff();
    mRotateSwitch->setAllChildrenOff();
    mManipulateSwitch->setAllChildrenOff();
    addChild(mMoveSwitch);
    addChild(mRotateSwitch);
    addChild(mManipulateSwitch);

    mMoveMatTransVector.clear();
    mRotateMatTransVector.clear();
    mManipulateMatTransVector.clear();
}


/***************************************************************
* Function: acceptCAVEGeodeShape()
*
* 'refShapeCenter' is reference center of 'CAVEGeodeShape', use
*  this vector as reference center in design object space, which
*  is associated with center of BoundingBox. Apply the offset of
* 'shapeCenter - refShapeCenter' on top level of scene graph and
*  then rescale them in lower levels to fit into the scale of 
*  surface icon space. 
*
***************************************************************/
void CAVEGroupEditGeodeWireframe::acceptCAVEGeodeShape(CAVEGeodeShape *shapeGeode, const osg::Vec3 &refShapeCenter)
{
    mRefShapeCenter = refShapeCenter;
    updateCAVEGeodeShape(shapeGeode);

    /* create 'MOVE' operational wireframe geode[0] */
    MatrixTransform *moveTrans = new MatrixTransform;
    CAVEGeodeEditWireframeMove *moveGeode = new CAVEGeodeEditWireframeMove;
    mMoveSwitch->addChild(moveTrans);
    mMoveMatTransVector.push_back(moveTrans);
    moveTrans->addChild(moveGeode);
    moveTrans->setMatrix(mBoundBoxScaleMat * mAccRootMat);

    /* create 'ROTATE' operational wireframe geode[0] */
    MatrixTransform *rotateTrans = new MatrixTransform;
    CAVEGeodeEditWireframeRotate *rotateGeode = new CAVEGeodeEditWireframeRotate;
    mRotateSwitch->addChild(rotateTrans);
    mRotateMatTransVector.push_back(rotateTrans);
    rotateTrans->addChild(rotateGeode);
    rotateTrans->setMatrix(mBoundSphereScaleMat * mAccRootMat);

    /* create 'MANIPULATE' operational wireframe geode[0] */
    MatrixTransform *manipulateTrans = new MatrixTransform;
    CAVEGeodeEditWireframeManipulate *manipulateGeode = new CAVEGeodeEditWireframeManipulate;
    mManipulateSwitch->addChild(manipulateTrans);
    mManipulateMatTransVector.push_back(manipulateTrans);
    manipulateTrans->addChild(manipulateGeode);
    manipulateTrans->setMatrix(mBoundBoxScaleMat * mAccRootMat);
}


/***************************************************************
* Function: updateCAVEGeodeShape()
*
* use bounding box of 'shapeGeode' to decide the sizes of wire
* frame and radius value used in rotation operations
*
***************************************************************/
void CAVEGroupEditGeodeWireframe::updateCAVEGeodeShape(CAVEGeodeShape *shapeGeode)
{
    const BoundingBox& bb = shapeGeode->getBoundingBox();

    float xmin, ymin, zmin, xmax, ymax, zmax, radius;
    xmin = bb.xMin();		xmax = bb.xMax();
    ymin = bb.yMin();		ymax = bb.yMax();
    zmin = bb.zMin();		zmax = bb.zMax();
    radius = bb.radius();

    mBoundingRadius = radius;
    mBoundBoxScaleMat.makeScale(Vec3(xmax-xmin, ymax-ymin, zmax-zmin));
    mBoundSphereScaleMat.makeScale(Vec3(radius, radius, radius));

    /* reset global shape geode offset, write initial value of 'mAccRootMat' */
    Vec3 shapeCenter = bb.center();
    mAccRootMat.makeTranslate(shapeCenter - mRefShapeCenter);
}


/***************************************************************
* Function: applyTranslation()
*
* 'gridSVect': number of snapping segments along each direction
* 'gridUnitLegnth': actual length represented by each segment,
*  only one component of 'gridSVect' is supposed to be non-zero
*
***************************************************************/
void CAVEGroupEditGeodeWireframe::applyTranslation(const osg::Vec3s &gridSVect, const float &gridUnitLegnth,
							const string &gridUnitLegnthInfo)
{
    mMoveSwitch->setAllChildrenOn();

    /* move to other direction: clear all children of 'mMoveSwitch' except child[0], rebuild offset tree */
    if ((mMoveSVect.x() * gridSVect.x() + mMoveSVect.y() * gridSVect.y() + mMoveSVect.z() * gridSVect.z()) <= 0)
    {
	unsigned int numChildren = mMoveSwitch->getNumChildren();
	if (numChildren > 1)
	{
	    mMoveSwitch->removeChildren(1, numChildren - 1);
	    MatrixTransVector::iterator itrMatTrans = mMoveMatTransVector.begin();
	    mMoveMatTransVector.erase(itrMatTrans + 1, itrMatTrans + numChildren);
	}
	mMoveSVect = Vec3s(0, 0, 0);
    }

    /* decide unit offset vector and start/end index of children under 'mMoveSwitch' */
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

    /* update the first wireframe with global translation and rotation */
    if (idxStart == 0) mMoveMatTransVector[0]->setMatrix(mBoundBoxScaleMat * mAccRootMat);

    /* create or remove a sequence of extra children under 'mMoveSwitch' */
    if (idxStart < idxEnd)
    {
	for (short i = idxStart + 1; i <= idxEnd; i++)
	{
	    Matrixd transMat;
	    transMat.makeTranslate(gridOffsetVect * i);
	    MatrixTransform *moveTrans = new MatrixTransform;
	    CAVEGeodeEditWireframeMove *moveGeode = new CAVEGeodeEditWireframeMove;
	    mMoveSwitch->addChild(moveTrans);
	    mMoveMatTransVector.push_back(moveTrans);
	    moveTrans->addChild(moveGeode);
	    moveTrans->setMatrix(mBoundBoxScaleMat * mAccRootMat * transMat);
	}
    }
    else if (idxStart > idxEnd)
    {
	mMoveSwitch->removeChildren(idxEnd + 1, idxStart - idxEnd);
	MatrixTransVector::iterator itrMatTrans = mMoveMatTransVector.begin();
	itrMatTrans += idxEnd + 1;
	mMoveMatTransVector.erase(itrMatTrans, itrMatTrans + (idxStart - idxEnd));
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
void CAVEGroupEditGeodeWireframe::applyRotation(const osg::Vec3s &axisSVect, const float &gridUnitAngle,
						const string &gridUnitAngleInfo)
{
    if (!mPrimaryFlag) return;

    mRotateSwitch->setAllChildrenOn();

    /* rotate in other direction: clear all children of 'mRotateSwitch' except child[0], rebuild offset tree */
    if ((mRotateSVect.x() * axisSVect.x() + mRotateSVect.y() * axisSVect.y() + mRotateSVect.z() * axisSVect.z()) <= 0)
    {
	unsigned int numChildren = mRotateSwitch->getNumChildren();
	if (numChildren > 1)
	{
	    mRotateSwitch->removeChildren(1, numChildren - 1);
	    MatrixTransVector::iterator itrMatTrans = mRotateMatTransVector.begin();
	    mRotateMatTransVector.erase(itrMatTrans + 1, itrMatTrans + numChildren);
	}
	mRotateSVect = Vec3s(0, 0, 0);
    }

    /* decide unit rotation quat and start/end index of children under 'mRotateSwitch' */
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

    /* create or remove a sequence of extra children under 'mRotateSwitch' */
    if (idxStart < idxEnd)
    {
	for (short i = idxStart + 1; i <= idxEnd; i++)
	{
	    Matrixd rotMat;
	    rotMat.makeRotate(gridUnitAngle * i, gridRotationAxis);
	    MatrixTransform *rotateTrans = new MatrixTransform;
	    CAVEGeodeEditWireframeRotate *rotateGeode = new CAVEGeodeEditWireframeRotate;
	    mRotateSwitch->addChild(rotateTrans);
	    mRotateMatTransVector.push_back(rotateTrans);
	    rotateTrans->addChild(rotateGeode);
	    rotateTrans->setMatrix(mBoundSphereScaleMat * rotMat);
	}
    }
    else if (idxStart > idxEnd)
    {
	mRotateSwitch->removeChildren(idxEnd + 1, idxStart - idxEnd);
	MatrixTransVector::iterator itrMatTrans = mRotateMatTransVector.begin();
	itrMatTrans += idxEnd + 1;
	mRotateMatTransVector.erase(itrMatTrans, itrMatTrans + (idxStart - idxEnd));
    }

    mRotateSVect = axisSVect;

    /* update info text if 'this' wireframe is primary */
    mEditInfoTextSwitch->setAllChildrenOn();

    char info[128];
    const float gridUnitAngleDegree = gridUnitAngle * 180 / M_PI;
    sprintf(info, "Angle = %3.2f\nSnapping = ", gridUnitAngleDegree * idxEnd);
    mEditInfoText->setText(info + gridUnitAngleInfo);
}


/***************************************************************
* Function: applyScaling()
*
* 'gridUnitScaleVect' is guranteed to be non-negative vector
*
***************************************************************/
void CAVEGroupEditGeodeWireframe::applyScaling( const short &nOffsetSegs, const Vec3 &gridUnitScaleVect, 
						const std::string &gridUnitScaleInfo)
{
    mManipulateSwitch->setAllChildrenOn();

    /* scale to other direction: clear all children of 'mManipulateSwitch' except child[0], rebuild offset tree */
    if (mScaleNumSegs * nOffsetSegs <= 0)
    {
	unsigned int numChildren = mManipulateSwitch->getNumChildren();
	if (numChildren > 1)
	{
	    mManipulateSwitch->removeChildren(1, numChildren - 1);
	    MatrixTransVector::iterator itrMatTrans = mManipulateMatTransVector.begin();
	    mManipulateMatTransVector.erase(itrMatTrans + 1, itrMatTrans + numChildren);
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

    /* update the first wireframe with global translation and rotation */
    if (idxStart == 0) mManipulateMatTransVector[0]->setMatrix(mBoundBoxScaleMat * mAccRootMat);

    /* create or remove a sequence of extra children under 'mMoveSwitch' */
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

	    Matrixd scaleMat;
	    scaleMat.makeScale(scaleVect);
	    MatrixTransform *scaleTrans = new MatrixTransform;
	    CAVEGeodeEditWireframeManipulate *manipulateGeode = new CAVEGeodeEditWireframeManipulate;
	    mManipulateSwitch->addChild(scaleTrans);
	    mManipulateMatTransVector.push_back(scaleTrans);
	    scaleTrans->addChild(manipulateGeode);
	    scaleTrans->setMatrix(mBoundBoxScaleMat * mAccRootMat * scaleMat);
	}
    }
    else if (idxStart > idxEnd)
    {
	mManipulateSwitch->removeChildren(idxEnd + 1, idxStart - idxEnd);
	MatrixTransVector::iterator itrMatTrans = mManipulateMatTransVector.begin();
	itrMatTrans += idxEnd + 1;
	mManipulateMatTransVector.erase(itrMatTrans, itrMatTrans + (idxStart - idxEnd));
    }

    mScaleNumSegs = nOffsetSegs;
    mScaleUnitVect = gridUnitScaleVect;

    if (!mPrimaryFlag) return;

    /* update info text if 'this' wireframe is primary */
    mEditInfoTextSwitch->setAllChildrenOn();

    float scaleUnit = gridUnitScaleVect.x() > 0 ? gridUnitScaleVect.x() : gridUnitScaleVect.y();
    scaleUnit = scaleUnit > 0 ? scaleUnit : gridUnitScaleVect.z();

    char info[128];
    float scaleval = scaleUnit * nOffsetSegs + 1.0f;
    scaleval = scaleval > 0 ? scaleval:0;
    sprintf(info, "Scale = %3.2f \nSnapping = ", scaleval);
    mEditInfoText->setText(info + gridUnitScaleInfo);
}


/***************************************************************
* Function: updateGridUnits()
***************************************************************/
void CAVEGroupEditGeodeWireframe::updateGridUnits(const float &gridUnitLegnth, 
			const float &gridUnitAngle, const float &gridUnitScale)
{
    /* update 'Move' matrix transform wireframes */
    unsigned int numMoveChildren = mMoveSwitch->getNumChildren();
    if (numMoveChildren > 1)
    {
	Vec3 gridOffsetVect = Vec3(0, 0, 0);
	if (mMoveSVect.x() > 0) gridOffsetVect = Vec3(gridUnitLegnth, 0, 0);
	else if (mMoveSVect.x() < 0) gridOffsetVect = Vec3(-gridUnitLegnth, 0, 0);
	else if (mMoveSVect.y() > 0) gridOffsetVect = Vec3(0,  gridUnitLegnth, 0);
	else if (mMoveSVect.y() < 0) gridOffsetVect = Vec3(0, -gridUnitLegnth, 0);
	else if (mMoveSVect.z() > 0) gridOffsetVect = Vec3(0, 0,  gridUnitLegnth);
	else if (mMoveSVect.z() < 0) gridOffsetVect = Vec3(0, 0, -gridUnitLegnth);

	for (int i = 0; i < numMoveChildren; i++)
	{
	    Matrixd transMat;
	    transMat.makeTranslate(gridOffsetVect * i);
	    mMoveMatTransVector[i]->setMatrix(mBoundBoxScaleMat * mAccRootMat * transMat);
	}
	return;
    }

    /* update 'Rotate' matrix transform wireframes */
    unsigned int numRotateChildren = mRotateSwitch->getNumChildren();
    if (numRotateChildren > 1)
    {
	Vec3 gridRotationAxis = Vec3(0, 0, 1);
	if (mRotateSVect.x() > 0) gridRotationAxis = Vec3(1, 0, 0);
	else if (mRotateSVect.x() < 0) gridRotationAxis = Vec3(-1, 0, 0);
	else if (mRotateSVect.y() > 0) gridRotationAxis = Vec3(0, 1, 0);
	else if (mRotateSVect.y() < 0) gridRotationAxis = Vec3(0, -1, 0);
	else if (mRotateSVect.z() > 0) gridRotationAxis = Vec3(0, 0, 1);
	else if (mRotateSVect.z() < 0) gridRotationAxis = Vec3(0, 0, -1);

	for (int i = 0; i < numRotateChildren; i++)
	{
	    Matrixd rotMat;
	    rotMat.makeRotate(gridUnitAngle * i, gridRotationAxis);
	    mRotateMatTransVector[i]->setMatrix(mBoundSphereScaleMat * rotMat);
	}
	return;
    }

    /* update 'Scale' matrix transform wireframes */
    unsigned int numScaleChildren = mManipulateSwitch->getNumChildren();
    if (numScaleChildren > 1)
    {
	/* rewrite scaling vector */
	mScaleUnitVect.x() = mScaleUnitVect.x() > 0 ? 1:0;
	mScaleUnitVect.y() = mScaleUnitVect.y() > 0 ? 1:0;
	mScaleUnitVect.z() = mScaleUnitVect.z() > 0 ? 1:0;
	mScaleUnitVect *= gridUnitScale;

	for (int i = 0; i < numScaleChildren; i++)
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
	    mManipulateMatTransVector[i]->setMatrix(mBoundBoxScaleMat * mAccRootMat * scaleMat);
	}
	return;
    }
}


/***************************************************************
* Function: applyEditorInfo()
***************************************************************/
void CAVEGroupEditGeodeWireframe::applyEditorInfo(CAVEGeodeShape::EditorInfo **infoPtr)
{
    /* reset root offset only when the shape is moved */
    if ((*infoPtr)->getTypeMasking() == CAVEGeodeShape::EditorInfo::MOVE)
    {
	/* apply translations on 'mAccRootMat' */
	const Vec3 offset = (*infoPtr)->getMoveOffset();

	Matrixd transMat;
	transMat.makeTranslate(offset);
	mAccRootMat = mAccRootMat * transMat;
    }
    else if ((*infoPtr)->getTypeMasking() == CAVEGeodeShape::EditorInfo::ROTATE)
    {
	/* apply rotations on 'mAccRootMat' */
	const float angle = (*infoPtr)->getRotateAngle();
	const Vec3 axis = (*infoPtr)->getRotateAxis();

	Matrixf rotateMat;
	rotateMat.makeRotate(angle, axis);
	mAccRootMat = mAccRootMat * rotateMat;
    }
    else if ((*infoPtr)->getTypeMasking() == CAVEGeodeShape::EditorInfo::SCALE)
    {
	const Vec3f scaleVect = (*infoPtr)->getScaleVect();
	const Vec3f scaleCenter = (*infoPtr)->getScaleCenter();

	Matrixd scalingMat, transMat, revTransMat;
	scalingMat.makeScale(scaleVect);
	transMat.makeTranslate(scaleCenter);
	revTransMat.makeTranslate(-scaleCenter);

	mAccRootMat = mAccRootMat * scalingMat;
    }
}


/***************************************************************
* Function: clearActiveWireframes()
***************************************************************/
void CAVEGroupEditGeodeWireframe::clearActiveWireframes()
{
    unsigned int numMoveChildren = mMoveSwitch->getNumChildren();
    if (numMoveChildren > 1)
    {
	mMoveSwitch->removeChildren(1, numMoveChildren - 1);
	MatrixTransVector::iterator itrMatTrans = mMoveMatTransVector.begin();
	mMoveMatTransVector.erase(itrMatTrans + 1, mMoveMatTransVector.end());
    }
    mMoveSVect = Vec3s(0, 0, 0);

    unsigned int numRotateChildren = mRotateSwitch->getNumChildren();
    if (numRotateChildren > 1)
    {
	mRotateSwitch->removeChildren(1, numRotateChildren - 1);
	MatrixTransVector::iterator itrMatTrans = mRotateMatTransVector.begin();
	mRotateMatTransVector.erase(itrMatTrans + 1, mRotateMatTransVector.end());
    }
    mRotateSVect = Vec3s(0, 0, 0);

    unsigned int numScaleChildren = mManipulateSwitch->getNumChildren();
    if (numScaleChildren > 1)
    {
	mManipulateSwitch->removeChildren(1, numScaleChildren - 1);
	MatrixTransVector::iterator itrMatTrans = mManipulateMatTransVector.begin();
	mManipulateMatTransVector.erase(itrMatTrans + 1, mManipulateMatTransVector.end());
    }
    mScaleNumSegs = 0;
    mScaleUnitVect = Vec3(0, 0, 0);

    /* turn off all active wireframes and text */
    mMoveSwitch->setAllChildrenOff();
    mRotateSwitch->setAllChildrenOff();
    mManipulateSwitch->setAllChildrenOff();
    mEditInfoTextSwitch->setAllChildrenOff();
}




