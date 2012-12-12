/***************************************************************
* File Name: DOGeometryEditor.cpp
*
* Description:
*
* Written by ZHANG Lelin on Jan 19, 2011
*
***************************************************************/
#include "DOGeometryEditor.h"


using namespace std;
using namespace osg;


// Constructor
DOGeometryEditor::DOGeometryEditor(): mLengthPerUnit(0.3048f), mAnglePerUnit(M_PI / 36.f), mScalePerUnit(0.2f),
		mInitSnapPos(Vec3(0, 0, 0)), mInitRadiusVector(Vec3(0, -1, 0)), mToolkitActiveIdx(-1), 
		mActiveIconToolkit(NULL), mDOGeometryCollectorPtr(NULL), mDOGeometryClonerPtr(NULL)
{
    mLengthPerUnitInfo = string("1 ft");
    mAnglePerUnitInfo = string("5 deg");
    mScalePerUnitInfo = string("0.2");

    mEditorInfo = new CAVEGeodeShape::EditorInfo();
}


/***************************************************************
* Function: initDesignObjects()
***************************************************************/
void DOGeometryEditor::initDesignObjects()
{
    /* load toolkit objects */
    CAVEAnimationModeler::ANIMLoadGeometryEditorIconToolkits(&mMatrixTrans,
			mNumToolkits, &mIconToolkitSwitchEntryArray);
    mDOIconToolkitSwitch->addChild(mMatrixTrans);
    mToolkitActiveIdx = 0;
}


/***************************************************************
* Function: setToolkitEnabled()
***************************************************************/
void DOGeometryEditor::setToolkitEnabled(bool flag)
{
    AnimationPathCallback* toolkitsAnimCallback = NULL;
    if (flag)
    {
        /* get translation hint from 'CAVEGroupIconSurface', apply offset to all icon groups */
        Vec3 iconCenter;
        CAVEGroupIconSurface::getSurfaceTranslationIconCenterHint(iconCenter);
        Matrixd transMat;
        transMat.makeTranslate(iconCenter);
        mMatrixTrans->setMatrix(transMat);

        mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mSwitch->setSingleChildOn(0);
        toolkitsAnimCallback = dynamic_cast <AnimationPathCallback*> 
            (mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mFwdAnimCallback);
    }
    else
    {
        mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mSwitch->setSingleChildOn(1);
        toolkitsAnimCallback = dynamic_cast <AnimationPathCallback*> 
            (mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mBwdAnimCallback);
    }
    setSubToolkitEnabled(flag);
    if (toolkitsAnimCallback) 
        toolkitsAnimCallback->reset();
}


/***************************************************************
* Function: setSubToolkitEnabled()
***************************************************************/
void DOGeometryEditor::setSubToolkitEnabled(bool flag)
{
    const CAVEGeodeIconToolkit::Type &toolkitTyp = (CAVEGeodeIconToolkit::Type) mToolkitActiveIdx;

    if (toolkitTyp == CAVEGeodeIconToolkit::CLONE)
    {
        if (flag) 
        {
            mDOGeometryClonerPtr->setToolkitEnabled(true);
        }
        else
        {
            mDOGeometryClonerPtr->popClonedObjects();
            mDOGeometryClonerPtr->setToolkitEnabled(false);
        }
    }
}


/***************************************************************
* Function: setPrevToolkit()
***************************************************************/
void DOGeometryEditor::setPrevToolkit()
{
    /* disable current toolkit icons */
    mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mSwitch->setSingleChildOn(1);
    AnimationPathCallback* pastCallback = dynamic_cast <AnimationPathCallback*> 
	(mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mBwdAnimCallback);
    if (pastCallback) 
        pastCallback->reset();

    setSubToolkitEnabled(false);
    if (--mToolkitActiveIdx < 0) 
        mToolkitActiveIdx = mNumToolkits - 1;
    setSubToolkitEnabled(true);

    /* enable new toolkit icons */
    mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mSwitch->setSingleChildOn(0);
    AnimationPathCallback* curCallback = dynamic_cast <AnimationPathCallback*> 
	(mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mFwdAnimCallback);
    if (curCallback) 
        curCallback->reset();
}


/***************************************************************
* Function: setNextToolkit()
***************************************************************/
void DOGeometryEditor::setNextToolkit()
{
    /* disable current toolkit icons */
    mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mSwitch->setSingleChildOn(1);
    AnimationPathCallback* pastCallback = dynamic_cast <AnimationPathCallback*> 
	(mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mBwdAnimCallback);
    if (pastCallback) 
        pastCallback->reset();

    setSubToolkitEnabled(false);
    if (++mToolkitActiveIdx >= mNumToolkits) 
        mToolkitActiveIdx = 0;
    setSubToolkitEnabled(true);

    /* enable new toolkit icons */
    mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mSwitch->setSingleChildOn(0);
    AnimationPathCallback* curCallback = dynamic_cast <AnimationPathCallback*> 
	(mIconToolkitSwitchEntryArray[mToolkitActiveIdx]->mFwdAnimCallback);
    if (curCallback) 
        curCallback->reset();
}


/***************************************************************
* Function: setScalePerUnit()
***************************************************************/
void DOGeometryEditor::setScalePerUnit(SnapLevelController **snapLevelControllerRefPtr)
{
    mLengthPerUnit = (*snapLevelControllerRefPtr)->getSnappingLength();
    mAnglePerUnit = (*snapLevelControllerRefPtr)->getSnappingAngle();
    mScalePerUnit = (*snapLevelControllerRefPtr)->getSnappingScale();

    mLengthPerUnitInfo = (*snapLevelControllerRefPtr)->getSnappingLengthInfo();
    mAnglePerUnitInfo = (*snapLevelControllerRefPtr)->getSnappingAngleInfo();
    mScalePerUnitInfo = (*snapLevelControllerRefPtr)->getSnappingScaleInfo();

    /* update editing wireframes and geometry offsets in realtime */
    applyGridUnitUpdates();
}


/***************************************************************
* Function: setSnapStarted()
***************************************************************/
void DOGeometryEditor::setSnapStarted(const osg::Vec3 &pos)
{
    mInitSnapPos = pos;

    /* compute 'mInitRadiusVector' each time when the pointer is pressed */
    Vec3 iconCenter;
    CAVEGroupIconSurface::getSurfaceTranslationIconCenterHint(iconCenter);
    mInitRadiusVector = pos - iconCenter;
    mInitRadiusVector.normalize();

    /* In 'COLLECT_GEODE' mode, 'CAVEGeode' objects under 'mDOShapeRefSwitch' are set on as an editing 
       indicator. In 'COLLECT_GEOMETRY', these objects are set off and used as geometry 'backup': data
       arrays in actual 'CAVEGeode' are applied with editing info with reference to these backup data. 
    */
    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
        mDOGeometryCollectorPtr->mDOShapeRefSwitch->setAllChildrenOn();
    }
    else if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEOMETRY)
    {
        mDOGeometryCollectorPtr->mDOShapeRefSwitch->setAllChildrenOff();
    }

    /* clone selected geometry the first time when 'CLONE' icon is clicked */
    if (!mActiveIconToolkit) 
        return;
    if (mActiveIconToolkit->getType() == CAVEGeodeIconToolkit::CLONE)
    {
        if (mDOGeometryClonerPtr->isToolkitEnabled())
        {
            mDOGeometryClonerPtr->pushClonedObjects();
            mDOGeometryClonerPtr->setToolkitEnabled(false);
        }
    }
}


/***************************************************************
* Function: setSnapUpdate()
***************************************************************/
void DOGeometryEditor::setSnapUpdate(const osg::Vec3 &pos)
{
    if (!mActiveIconToolkit) return;

    Vec3 offset = pos - mInitSnapPos;

    /* translate offset value with respect to MOVE or CLONE toolkit */
    if (mActiveIconToolkit->getType() == CAVEGeodeIconToolkit::MOVE ||
        mActiveIconToolkit->getType() == CAVEGeodeIconToolkit::CLONE)
    {
        CAVEGeodeIconToolkitMove *toolkitMove = dynamic_cast <CAVEGeodeIconToolkitMove*> (mActiveIconToolkit);
        if (!toolkitMove) 
            return;

        CAVEGeodeIconToolkitMove::FaceOrientation faceorient = toolkitMove->getFaceOrientation();
        switch (faceorient)
        {
            case CAVEGeodeIconToolkitMove::FRONT_BACK: 
            {
                applyTranslation(Vec3(0, offset.y(), 0));
                break;
            }
            case CAVEGeodeIconToolkitMove::LEFT_RIGHT: 
            {
                applyTranslation(Vec3(offset.x(), 0, 0)); 
                break; 
            }
            case CAVEGeodeIconToolkitMove::UP_DOWN: 
            {
                applyTranslation(Vec3(0, 0, offset.z())); 
                break;
            }
            default: 
            {
                break;
            }
        }
    }

    /* translate offset value with respect to ROTATE toolkit */
    else if (mActiveIconToolkit->getType() == CAVEGeodeIconToolkit::ROTATE)
    {
        CAVEGeodeIconToolkitRotate *toolkitRotate = dynamic_cast <CAVEGeodeIconToolkitRotate*> (mActiveIconToolkit);
        if (!toolkitRotate) 
            return;

        CAVEGeodeIconToolkitRotate::AxisAlignment axisalignment = toolkitRotate->getAxisAlignment();
        switch (axisalignment)
        {
            case CAVEGeodeIconToolkitRotate::X_AXIS: applyRotation(Vec3(1, 0, 0), offset); break;
            case CAVEGeodeIconToolkitRotate::Y_AXIS: applyRotation(Vec3(0, 1, 0), offset); break; 
            case CAVEGeodeIconToolkitRotate::Z_AXIS: applyRotation(Vec3(0, 0, 1), offset); break; 
            default: break;
        }
    }

    /* translate offset value with respect to MANIPULATE toolkit */
    else if (mActiveIconToolkit->getType() == CAVEGeodeIconToolkit::MANIPULATE)
    {
        CAVEGeodeIconToolkitManipulate *toolkitManipulate = 
            dynamic_cast <CAVEGeodeIconToolkitManipulate*> (mActiveIconToolkit);
        if (!toolkitManipulate) 
            return;

        applyManipulation(toolkitManipulate->getScalingDir(), toolkitManipulate->getBoundingVect(), offset);
    }
}


/***************************************************************
* Function: setSnapFinished()
***************************************************************/
void DOGeometryEditor::setSnapFinished()
{
    applyEditingFinishes();
    mEditorInfo->reset();

    /* set all reference 'CAVEGeodeShape' invisible, reset transform matrix */
    mDOGeometryCollectorPtr->mDOShapeRefSwitch->setAllChildrenOff();
    Matrixd identityMat;
    mDOGeometryCollectorPtr->mDOShapeRefMatrixTrans->setMatrix(identityMat);
}


/***************************************************************
* Function: setPointerDir()
***************************************************************/
void DOGeometryEditor::setPointerDir(const Vec3 &pointerDir)
{
    CAVEGroupEditWireframe::setPointerDir(pointerDir);

    /* update osg text3D orientations based on pointer front direction */
    CAVEGroupEditGeodeWireframeVector& geodeWireframeVector = mDOGeometryCollectorPtr->getEditGeodeWireframeVector();
    CAVEGroupEditGeodeWireframeVector::iterator itrGeodeWireframe;
    if (geodeWireframeVector.size() > 0)
    {
        for (itrGeodeWireframe = geodeWireframeVector.begin(); 
             itrGeodeWireframe != geodeWireframeVector.end();
             itrGeodeWireframe++) 
        {
            (*itrGeodeWireframe)->resetInfoOrientation();
        }
    }

    CAVEGroupEditGeometryWireframeVector& geomWireframeVector = mDOGeometryCollectorPtr->getEditGeometryWireframeVector();
    CAVEGroupEditGeometryWireframeVector::iterator itrGeomWireframe;
    if (geomWireframeVector.size() > 0)
    {
        for (itrGeomWireframe = geomWireframeVector.begin(); 
             itrGeomWireframe != geomWireframeVector.end();
             itrGeomWireframe++) 
        {
            (*itrGeomWireframe)->resetInfoOrientation();
        }
    }
}


/***************************************************************
* Function: setActiveIconToolkit()
***************************************************************/
void DOGeometryEditor::setActiveIconToolkit(CAVEGeodeIconToolkit *iconToolkit)
{
    if (iconToolkit) 
        iconToolkit->pressed();
    else if (mActiveIconToolkit) 
        mActiveIconToolkit->released();

    mActiveIconToolkit = iconToolkit;
}


/***************************************************************
* Function: setDOGeometryCollectorPtr()
***************************************************************/
void DOGeometryEditor::setDOGeometryCollectorPtr(DOGeometryCollector *geomCollectorPtr)
{
    mDOGeometryCollectorPtr = geomCollectorPtr;
}


/***************************************************************
* Function: setDOGeometryClonerPtr()
***************************************************************/
void DOGeometryEditor::setDOGeometryClonerPtr(DOGeometryCloner *geomClonerPtr)
{
    mDOGeometryClonerPtr = geomClonerPtr;
}


/***************************************************************
* Function: applyTranslation()
***************************************************************/
void DOGeometryEditor::applyTranslation(const osg::Vec3 &offset)
{
    /* calculate numbers of snapping units along 'offset' and the actual length it represents for */
    const float unit = CAVEGeodeEditWireframe::gSnappingUnitDist;
    short nSegX, nSegY, nSegZ;
    if (offset.x() > 0) 
        nSegX = (short)(offset.x() / unit + 0.5f);
    else 
        nSegX = (short)(offset.x() / unit - 0.5f);

    if (offset.y() > 0) 
        nSegY = (short)(offset.y() / unit + 0.5f);
    else 
        nSegY = (short)(offset.y() / unit - 0.5f);

    if (offset.z() > 0) 
        nSegZ = (short)(offset.z() / unit + 0.5f);
    else 
        nSegZ = (short)(offset.z() / unit - 0.5f);

    const Vec3s gridIntVect = Vec3s(nSegX, nSegY, nSegZ);

    /* update translation portion of 'mEditorInfo' */
    mEditorInfo->setMoveUpdate(Vec3(nSegX, nSegY, nSegZ) * mLengthPerUnit);

    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
        /* update wireframe vector */
        CAVEGroupEditGeodeWireframeVector& geodeWireframeVector = mDOGeometryCollectorPtr->getEditGeodeWireframeVector();
        CAVEGroupEditGeodeWireframeVector::iterator itrGeodeWireframe;
        if (geodeWireframeVector.size() > 0)
        {
            for (itrGeodeWireframe = geodeWireframeVector.begin(); 
                 itrGeodeWireframe != geodeWireframeVector.end();
                 itrGeodeWireframe++) 
            {
                (*itrGeodeWireframe)->applyTranslation(gridIntVect, mLengthPerUnit, mLengthPerUnitInfo);
            }
        }

        /* reflect temporary editing effects on 'mDOShapeRefMatrixTrans' */
        Matrixd transMat;
        transMat.makeTranslate(Vec3(nSegX, nSegY, nSegZ) * mLengthPerUnit);
        mDOGeometryCollectorPtr->mDOShapeRefMatrixTrans->setMatrix(transMat);
    }
    else if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEOMETRY)
    {
        /* update wireframe vector */
        CAVEGroupEditGeometryWireframeVector &geomWireframeVector = 
            mDOGeometryCollectorPtr->getEditGeometryWireframeVector();
        CAVEGroupEditGeometryWireframeVector::iterator itrGeomWireframe;
        if (geomWireframeVector.size() > 0)
        {
            for (itrGeomWireframe = geomWireframeVector.begin(); 
                 itrGeomWireframe != geomWireframeVector.end();
                 itrGeomWireframe++)
            {
                (*itrGeomWireframe)->applyTranslation(gridIntVect, mLengthPerUnit, mLengthPerUnitInfo);
            }
        }

        /* reflect temporary editing effects on 'CAVEGeodeShapes' */
        applyEditingUpdates();
    }
}


/***************************************************************
* Function: applyRotation()
***************************************************************/
void DOGeometryEditor::applyRotation(const osg::Vec3 &axis, const osg::Vec3 &offset)
{
    /* compute number of offset grids around axis: only works for rotations around X, Y, Z axis */
    const float unit = CAVEGeodeEditWireframe::gSnappingUnitDist;
    const float len = (offset ^ axis) * mInitRadiusVector;
    short nGrids = 0;
    if (len > 0) 
        nGrids = (short)(len / unit + 0.5f);
    else 
        nGrids = (short)(len / unit - 0.5f);

    Vec3s axisSVect = Vec3s((short)axis.x(), (short)axis.y(), (short)axis.z()) * nGrids;

    /* update rotation portion of 'mEditorInfo' */
    mEditorInfo->setRotateUpdate(mDOGeometryCollectorPtr->mShapeCenter, axis, nGrids * mAnglePerUnit);

    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
        CAVEGroupEditGeodeWireframeVector& geodeWireframeVector = mDOGeometryCollectorPtr->getEditGeodeWireframeVector();
        CAVEGroupEditGeodeWireframeVector::iterator itrGeodeWireframe;
        if (geodeWireframeVector.size() > 0)
        {
            for (itrGeodeWireframe = geodeWireframeVector.begin(); 
                 itrGeodeWireframe != geodeWireframeVector.end();
                 itrGeodeWireframe++) 
            {
                (*itrGeodeWireframe)->applyRotation(axisSVect, mAnglePerUnit, mAnglePerUnitInfo);
            }
        }

        /* reflect temporary editing effects on 'mDOShapeRefMatrixTrans' */
        const Vec3 rotCenter = mDOGeometryCollectorPtr->mShapeCenter;
        Matrixd rotateMat, transMat, revTransMat;
        rotateMat.makeRotate(mAnglePerUnit * nGrids, axis);
        transMat.makeTranslate(rotCenter);
        revTransMat.makeTranslate(-rotCenter);
        mDOGeometryCollectorPtr->mDOShapeRefMatrixTrans->setMatrix(revTransMat * rotateMat * transMat);
    }
    else if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEOMETRY)
    {
        /* update wireframe vector */
        CAVEGroupEditGeometryWireframeVector &geomWireframeVector = 
            mDOGeometryCollectorPtr->getEditGeometryWireframeVector();
        CAVEGroupEditGeometryWireframeVector::iterator itrGeomWireframe;
        if (geomWireframeVector.size() > 0)
        {
            for (itrGeomWireframe = geomWireframeVector.begin();    
                 itrGeomWireframe != geomWireframeVector.end();
                 itrGeomWireframe++)
            {
                (*itrGeomWireframe)->applyRotation(axisSVect, mAnglePerUnit, mAnglePerUnitInfo);
            }
        }

        /* reflect temporary editing effects on 'CAVEGeodeShapes' */
        applyEditingUpdates();
    }
}


/***************************************************************
* Function: applyManipulation()
***************************************************************/
void DOGeometryEditor::applyManipulation(const osg::Vec3 &dir, const osg::Vec3 &bound, const osg::Vec3 &offset)
{
    Vec3 dirControlPoint = Vec3(bound.x() * dir.x(), bound.y() * dir.y(), bound.z() * dir.z());
    dirControlPoint.normalize();
    Vec3 dirabs = Vec3(	(dir.x() > 0? dir.x(): -dir.x()), 
			(dir.y() > 0? dir.y(): -dir.y()), 
			(dir.z() > 0? dir.z(): -dir.z()));

    /* scalar value: offset vector projected on 'dirControlPoint' */
    const float unit = CAVEGeodeEditWireframe::gSnappingUnitDist;
    float offsetval = offset * dirControlPoint;
    short nOffsetSegs = 0;
    if (offsetval > 0) 
        nOffsetSegs = (short)(offsetval / unit + 0.5f);
    else 
        nOffsetSegs = (short)(offsetval / unit - 0.5f);
    const float scaleVal = nOffsetSegs * mScalePerUnit;

    /* update scaling portion of 'mEditorInfo' */
    const Vec3 scaleCenter = mDOGeometryCollectorPtr->mShapeCenter;
    Vec3 scaleVect = dirabs * scaleVal + Vec3(1, 1, 1);
    scaleVect.x() = scaleVect.x() > 0 ? scaleVect.x() : 0;
    scaleVect.y() = scaleVect.y() > 0 ? scaleVect.y() : 0;
    scaleVect.z() = scaleVect.z() > 0 ? scaleVect.z() : 0;
    mEditorInfo->setScaleUpdate(scaleCenter, scaleVect);

    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
        CAVEGroupEditGeodeWireframeVector& geodeWireframeVector = mDOGeometryCollectorPtr->getEditGeodeWireframeVector();
        CAVEGroupEditGeodeWireframeVector::iterator itrGeodeWireframe;
        if (geodeWireframeVector.size() > 0)
        {
            for (itrGeodeWireframe = geodeWireframeVector.begin(); 
                 itrGeodeWireframe != geodeWireframeVector.end();
                 itrGeodeWireframe++) 
            {
                (*itrGeodeWireframe)->applyScaling(nOffsetSegs, dirabs * mScalePerUnit, mScalePerUnitInfo);
            }
        }

        /* reflect temporary editing effects on 'mDOShapeRefMatrixTrans' */
        Matrixd scalingMat, transMat, revTransMat;
        scalingMat.makeScale(scaleVect);
        transMat.makeTranslate(scaleCenter);
        revTransMat.makeTranslate(-scaleCenter);
        mDOGeometryCollectorPtr->mDOShapeRefMatrixTrans->setMatrix(revTransMat * scalingMat * transMat);
    }
    else if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEOMETRY)
    {
        /* update wireframe vector */
        CAVEGroupEditGeometryWireframeVector &geomWireframeVector = 
            mDOGeometryCollectorPtr->getEditGeometryWireframeVector();
        CAVEGroupEditGeometryWireframeVector::iterator itrGeomWireframe;
        if (geomWireframeVector.size() > 0)
        {
            for (itrGeomWireframe = geomWireframeVector.begin(); 
                 itrGeomWireframe != geomWireframeVector.end();
                 itrGeomWireframe++)
            {
                (*itrGeomWireframe)->applyScaling(nOffsetSegs, dirabs * mScalePerUnit, mScalePerUnitInfo);
            }
        }

        /* reflect temporary editing effects on 'CAVEGeodeShapes' */
        applyEditingUpdates();
    }
}


/***************************************************************
* Function: applyGridUnitUpdates()
***************************************************************/
void DOGeometryEditor::applyGridUnitUpdates()
{
    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
        /* reset inter distances & offsets between geode wireframes */
        CAVEGroupEditGeodeWireframeVector& geodeWireframeVector = mDOGeometryCollectorPtr->getEditGeodeWireframeVector();
        CAVEGroupEditGeodeWireframeVector::iterator itrGeodeWireframe;
        if (geodeWireframeVector.size() > 0)
        {
            for (itrGeodeWireframe = geodeWireframeVector.begin(); 
                 itrGeodeWireframe != geodeWireframeVector.end();
                 itrGeodeWireframe++) 
            {
                (*itrGeodeWireframe)->updateGridUnits(mLengthPerUnit, mAnglePerUnit, mScalePerUnit);
            }
        }
    }
    else if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEOMETRY)
    {
        /* reset inter distances & offsets between geometry wireframes */
        CAVEGroupEditGeometryWireframeVector &geomWireframeVector = 
            mDOGeometryCollectorPtr->getEditGeometryWireframeVector();
        CAVEGroupEditGeometryWireframeVector::iterator itrGeomWireframe;
        if (geomWireframeVector.size() > 0)
        {
            for (itrGeomWireframe = geomWireframeVector.begin(); 
                 itrGeomWireframe != geomWireframeVector.end();
                 itrGeomWireframe++)
            {
                (*itrGeomWireframe)->updateGridUnits(mLengthPerUnit, mAnglePerUnit, mScalePerUnit);
            }
        }
    }
}


/***************************************************************
* Function: applyEditingUpdates()
*
* This function is only called during geometry level editing:
* Geometries in 'CAVEGeodeShape' objects under 'mDOShapeRefMatrixTrans'
* are used constant references to those collected in 'mGeodeShapeVector',
* temporal editing effects are applied to collected 'CAVEGeodeShape'
* objects in each snapping update. Final editing effects are
* applied to children of 'mDOShapeRefMatrixTrans' when snapping
* is finished.  
*
***************************************************************/
void DOGeometryEditor::applyEditingUpdates()
{
    if (mDOGeometryCollectorPtr->mMode != DOGeometryCollector::COLLECT_GEOMETRY) return;

    CAVEGeodeShapeVector& geodeShapeVector = mDOGeometryCollectorPtr->getGeodeShapeVector();
    const int nGeodes = geodeShapeVector.size();
    if (nGeodes > 0)
    {
	for (int i = 0; i < nGeodes; i++)
	{
	    CAVEGeodeShape *orgGeode = geodeShapeVector[i];		// get collected original 'CAVEGeodeShape'
	    CAVEGeodeShape *refGeode = dynamic_cast <CAVEGeodeShape*> 	// get cloned reference 'CAVEGeodeShape'
		(mDOGeometryCollectorPtr->mDOShapeRefMatrixTrans->getChild(i));
	    if (refGeode) orgGeode->applyEditorInfo(&mEditorInfo, refGeode);
	}
    }
}


/***************************************************************
* Function: applyEditingFinishes()
*
* Turn off all active wireframes, apply final editing effects
* to both reference geometry and actual geode shapes
*
***************************************************************/
void DOGeometryEditor::applyEditingFinishes()
{
    CAVEGroupEditGeodeWireframeVector& geodeWireframeVector = mDOGeometryCollectorPtr->getEditGeodeWireframeVector();
    const int nGeodeWireframes = geodeWireframeVector.size();

    CAVEGroupEditGeometryWireframeVector& geomWireframeVector = mDOGeometryCollectorPtr->getEditGeometryWireframeVector();
    const int nGeometryWireframes = geomWireframeVector.size();		

    CAVEGeodeShapeVector& geodeShapeVector = mDOGeometryCollectorPtr->getGeodeShapeVector();
    const int nGeodes = geodeShapeVector.size();

    /* apply editing info to 'GroupIconSurface' as updated representation of actual geodes */
    CAVEGroupIconSurfaceVector& groupIconSurfaceVector = mDOGeometryCollectorPtr->getGroupIconSurfaceVector();
    if (groupIconSurfaceVector.size() > 0)
    {
        for (int i = 0; i < groupIconSurfaceVector.size(); i++) 
        {
            groupIconSurfaceVector[i]->applyEditingFinishes(&mEditorInfo);
        }
    }

    /* unified application of editor info to two sets of 'CAVEGeodes', with reference to vertex masking array */
    if (nGeodes > 0)
    {
        for (int i = 0; i < nGeodes; i++)
        {
            CAVEGeodeShape *refGeode = dynamic_cast <CAVEGeodeShape*>
            (mDOGeometryCollectorPtr->mDOShapeRefMatrixTrans->getChild(i));
            if (refGeode)
            {
                geodeShapeVector[i]->applyEditorInfo(&mEditorInfo, refGeode);	// modify selected CAVEGeode first
                refGeode->applyEditorInfo(&mEditorInfo);			// modify cloned reference CAVEGeode
            }
        }   
    }

    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
        /* 'groupIconSurfaceVector', 'geodeWireframeVector', 'geodeShapeVector' and 'mDOShapeRefMatrixTrans' 
            must have the same size through editing process. After each snapping step, indicator geodes, 
            actual geodes and references geodes will be modified in parallel.
         */
        if (nGeodeWireframes > 0)
        {
            for (int i = 0; i < nGeodeWireframes; i++)
            {
                geodeWireframeVector[i]->applyEditorInfo(&mEditorInfo);
                geodeWireframeVector[i]->clearActiveWireframes();
            }
        }

        /* use the first geode in 'mGeodeShapeVector' as hint to update manipulator sizes, notice that there is 
           a chance that the first geode is not the same as the first selected geode, thus, the manipulator might
           be resized with 'bb.radius()' of the first selected geode and 'bb' of the first geode in the vector. 
           This is not strictly following the visual resizing rules, however, does not affect its usability.
        */
        CAVEGroupIconToolkit::setManipulatorBound(mDOGeometryCollectorPtr->mGeodeShapeVector[0]->getBoundingBox());
    }
    else if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEOMETRY)
    {
        /* 'geomWireframeVector' and 'geometryVector' sizes the same with respect to number of active surfaces. */
        if (nGeometryWireframes > 0)
        {
            for (int i = 0; i < nGeometryWireframes; i++)
            {
                geomWireframeVector[i]->updateCAVEGeometry();
                geomWireframeVector[i]->clearActiveWireframes();
            }
        }

        /* bounding boxes of geode wierframes are subject to changes in geometry level editing */
        if (nGeodeWireframes > 0)
        {
            for (int i = 0; i < nGeodeWireframes; i++)
            {
                geodeWireframeVector[i]->updateCAVEGeodeShape(geodeShapeVector[i]);
            }
        }
    }
}

