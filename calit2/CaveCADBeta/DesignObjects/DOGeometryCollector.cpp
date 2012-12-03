/***************************************************************
* File Name: DOGeometryCollector.cpp
*
* Description:
*
* Written by ZHANG Lelin on Jan 19, 2011
*
***************************************************************/
#include "DOGeometryCollector.h"


using namespace std;
using namespace osg;


// Constructor
DOGeometryCollector::DOGeometryCollector(): mMode(COLLECT_NONE), mShapeCenter(Vec3(0, 0, 0)), mIconCenter(Vec3(0, 0, 0))
{
    mGroupIconSurfaceVector.clear();
    mEditGeodeWireframeVector.clear();
    mEditGeometryWireframeVector.clear();

    mGeodeShapeVector.clear();
    mGeometryVector.clear();
}


/***************************************************************
* Function: initDesignObjects()
***************************************************************/
void DOGeometryCollector::initDesignObjects()
{
    /* create root for reference CAVEGeodeShape objects */
    mDOShapeRefSwitch = new Switch;
    mDOShapeRefMatrixTrans = new MatrixTransform;

    mNonIntersectableSceneGraphPtr->addChild(mDOShapeRefSwitch);
    mDOShapeRefSwitch->addChild(mDOShapeRefMatrixTrans);
    mDOShapeRefSwitch->setAllChildrenOff();

    /* create switches for wireframe display */
    mDOGeodeWireframeSwitch = new Switch;
    mDOGeometryWireframeSwitch = new Switch;
    mNonIntersectableSceneGraphPtr->addChild(mDOGeodeWireframeSwitch);
    mNonIntersectableSceneGraphPtr->addChild(mDOGeometryWireframeSwitch);
}


/***************************************************************
* Function: setSurfaceCollectionHints()
***************************************************************/
void DOGeometryCollector::setSurfaceCollectionHints(CAVEGeodeShape *shapeGeode, const osg::Vec3 &iconCenter)
{
    mShapeCenter = shapeGeode->getCAVEGeodeCenter();
    mIconCenter = iconCenter;

    /* set origin of the two coordinate systems: using center of first hit 'CAVEGeodeShape' and 
       center of current design state sphere. */
    CAVEGroupIconSurface::setSurfaceScalingHint(shapeGeode->getBoundingBox());
    CAVEGroupIconSurface::setSurfaceTranslationHint(mShapeCenter, mIconCenter);

    /* set offset distance of control points at corners of the selected shape */
    CAVEGroupIconToolkit::setManipulatorBoundRadius(shapeGeode->getBoundingBox());
    CAVEGroupIconToolkit::setManipulatorBound(shapeGeode->getBoundingBox());
}


/***************************************************************
* Function: toggleCAVEGeodeShape()
*
* This function is called each time when a CAVEGeodeShape is hit
*
***************************************************************/
void DOGeometryCollector::toggleCAVEGeodeShape(CAVEGeodeShape *shapeGeode)
{
    int vectorIdx = shapeGeode->mDOCollectorIndex;

    /* mDOCollectorIndex = -1, 'shapeGeode' has not been selected, push it to vector and
       assign an index value associates with its position in 'mGeodeShapeVector' */
    if (vectorIdx < 0)
    {
        /* assign an incremental collector index */
        shapeGeode->mDOCollectorIndex = mGeodeShapeVector.size();
        mGeodeShapeVector.push_back(shapeGeode);

        /* make a instance clone of toggled geode, asign the same vector index to it */
        CAVEGeodeShape *shapeGeodeRef = new CAVEGeodeShape(shapeGeode);
        shapeGeodeRef->mDOCollectorIndex = shapeGeode->mDOCollectorIndex;
        shapeGeodeRef->applyAlpha(0.5f);
        mDOShapeRefMatrixTrans->addChild(shapeGeodeRef);

        /* create icon surfaces and toolkits objects & animations at central position of state sphere:
           1) generate a vector of 'CAVEGeodeIconSurface' objects and add them to 'surfacesPATrans'
           2) add 'surfacesPATrans' to 'mDOIconSurfaceSwitch'
           3) reset pick up animations for surface icon
        */
        PositionAttitudeTransform *surfacesPATrans;
        CAVEGroupIconSurface *iconSurfaceGroup;
        CAVEAnimationModeler::ANIMLoadGeometryCollectorIconSurfaces(&surfacesPATrans, &iconSurfaceGroup, 
                                        shapeGeode, shapeGeodeRef);
        mGroupIconSurfaceVector.push_back(iconSurfaceGroup);
        mDOIconSurfaceSwitch->addChild(surfacesPATrans);

        AnimationPathCallback* surfacesAnimCallback = dynamic_cast <AnimationPathCallback*>
            (surfacesPATrans->getUpdateCallback());
        if (surfacesAnimCallback) 
            surfacesAnimCallback->reset();

        /* create geode wireframe, add it to reference switch and push back to wireframe vector */
        MatrixTransform *wireframeTrans;
        CAVEGroupEditGeodeWireframe *editGeodeWireframe;
        CAVEAnimationModeler::ANIMLoadGeometryCollectorGeodeWireframe(&wireframeTrans, &editGeodeWireframe, shapeGeode);
        mEditGeodeWireframeVector.push_back(editGeodeWireframe);
        mDOGeodeWireframeSwitch->addChild(wireframeTrans);

        /* mark 'editGeodeWireframe' as primary wireframe group if this geode shape is the first one selected */
        if (shapeGeode->mDOCollectorIndex == 0) 
            editGeodeWireframe->setPrimaryFlag(true);
    }

    /* mDOCollectorIndex > 0, 'shapeGeode' has already been selected, erase it from vector
       reset its index value, modify all other index values of the Geodes after this shape
       in the vector by decreasing 1. Also, clear all surfaces of this geode that has been
       selected in the vector. */
    else
    {
        /* remove the clone instance from 'mDOShapeRefMatrixTrans' */
        mDOShapeRefMatrixTrans->removeChild(shapeGeode->mDOCollectorIndex);

        mGeodeShapeVector.erase(mGeodeShapeVector.begin() + vectorIdx);
        mEditGeodeWireframeVector.erase(mEditGeodeWireframeVector.begin() + vectorIdx);
        mGroupIconSurfaceVector.erase(mGroupIconSurfaceVector.begin() + vectorIdx);

        mDOGeodeWireframeSwitch->removeChild(shapeGeode->mDOCollectorIndex);
        mDOIconSurfaceSwitch->removeChild(shapeGeode->mDOCollectorIndex);
        shapeGeode->mDOCollectorIndex = -1;

        /* 'mDOCollectorIndex' is a hint to trace all object instances in 'mGeodeShapeVector', 
           'mEditGeodeWireframeVector' and 'mGroupIconSurfaceVector'. Need to be re-ordered after erase operation. */
        if (mGeodeShapeVector.size() > 0)
        {
            CAVEGeodeShapeVector::iterator itrGeodeShape;
            for (itrGeodeShape = mGeodeShapeVector.begin() + vectorIdx; itrGeodeShape != mGeodeShapeVector.end();
             itrGeodeShape++)
            (*itrGeodeShape)->mDOCollectorIndex--;
        }
    }

    if (mGeodeShapeVector.size() > 0) 
        mMode = COLLECT_GEODE;
    else 
        mMode = COLLECT_NONE;

    updateVertexMaskingVector();
}


/***************************************************************
* Function: toggleCAVEGeodeIconSurface()
***************************************************************/
void DOGeometryCollector::toggleCAVEGeodeIconSurface(CAVEGeodeIconSurface *iconGeodeSurface)
{
    /* get original CAVEGeometry pointer which 'iconGeodeSurface' refers to */
    CAVEGeometry *orgGeometry = iconGeodeSurface->getGeometryOriginPtr();
    CAVEGeometry *refGeometry = iconGeodeSurface->getGeometryReferencePtr();
    int vectorIdx = orgGeometry->mDOCollectorIndex;

    /* mDOCollectorIndex = -1, 'orgGeometry' has not been selected, push it to vector and
       assign an index value associates with its position in 'mGeometryVector' */
    if (vectorIdx < 0)
    {
        /* assign an incremental collector index */
        orgGeometry->mDOCollectorIndex = mGeometryVector.size();
        mGeometryVector.push_back(orgGeometry);

        /* set highlight of 'iconGeodeSurface' as 'selected' */
        iconGeodeSurface->setHighlightSelected();

        /* create geometry wireframe, add it to reference switch and push back to wireframe vector */
        MatrixTransform *wireframeTrans;
        CAVEGroupEditGeometryWireframe *editGeometryWireframe;
        CAVEAnimationModeler::ANIMLoadGeometryCollectorGeometryWireframe(&wireframeTrans, &editGeometryWireframe, 										refGeometry);
        mEditGeometryWireframeVector.push_back(editGeometryWireframe);
        mDOGeometryWireframeSwitch->addChild(wireframeTrans);

        /* mark 'editGeometryWireframe' as primary wireframe group if this geometry is the first one selected */
        if (orgGeometry->mDOCollectorIndex == 0) 
            editGeometryWireframe->setPrimaryFlag(true);
    }

    /* mDOCollectorIndex > 0, 'orgGeometry' has already been selected, erase it from vector
       reset its index value, modify all other index values of CAVEGeometry after it in vector
       by decreasing 1. */
    else
    {
        /* set highlight of 'iconGeodeSurface' as 'unselected' */
        iconGeodeSurface->setHighlightUnselected();

        mGeometryVector.erase(mGeometryVector.begin() + vectorIdx);
        mEditGeometryWireframeVector.erase(mEditGeometryWireframeVector.begin() + vectorIdx);
        mDOGeometryWireframeSwitch->removeChild(vectorIdx);
        orgGeometry->mDOCollectorIndex = -1;

        if (mGeometryVector.size() > 0)
        {
            CAVEGeometryVector::iterator itrGeometry;
            for (itrGeometry = mGeometryVector.begin() + vectorIdx; itrGeometry != mGeometryVector.end(); itrGeometry++)
            (*itrGeometry)->mDOCollectorIndex--;
        }
    }

    if (mGeometryVector.size() > 0) mMode = COLLECT_GEOMETRY;
    else mMode = COLLECT_GEODE;
    updateVertexMaskingVector();
}


/***************************************************************
* Function: isGeodeShapeVectorEmpty()
***************************************************************/
bool DOGeometryCollector::isGeodeShapeVectorEmpty()
{
    if (mGeodeShapeVector.size() == 0) return true;
    return false;
}


/***************************************************************
* Function: ismGeometryVectorEmpty()
***************************************************************/
bool DOGeometryCollector::isGeometryVectorEmpty()
{
    if (mGeometryVector.size() == 0) return true;
    return false;
}


/***************************************************************
* Function: clearGeometryVector()
***************************************************************/
void DOGeometryCollector::clearGeometryVector()
{
    /* clear 'mDOCollectorIndex' of all 'CAVEGeometry' objects that have been selected */
    if (mGeometryVector.size() > 0)
    {
        CAVEGeometryVector::iterator itrGeometry;
        for (itrGeometry = mGeometryVector.begin(); itrGeometry != mGeometryVector.end(); itrGeometry++)
        {
            (*itrGeometry)->mDOCollectorIndex = -1;

            /* set highlights of all selected surfaces in 'mGeometryVector' as normal */
            setAllIconSurfaceHighlightsNormal();
        }
        mMode = COLLECT_GEODE;
        updateVertexMaskingVector();
    }

    mEditGeometryWireframeVector.clear();
    mDOGeometryWireframeSwitch->removeChildren(0, mGeometryVector.size());
    mGeometryVector.clear();
}


/***************************************************************
* Function: clearGeodeShapeVector()
***************************************************************/
void DOGeometryCollector::clearGeodeShapeVector()
{
    /* remove all 'CAVEGeodeShape' instances on an empty click of the input device */
    int numChildrenToRemove = mDOShapeRefMatrixTrans->getNumChildren();
    if (numChildrenToRemove > 0) 
        mDOShapeRefMatrixTrans->removeChildren(0, numChildrenToRemove);

    /* clear 'mDOCollectorIndex' of all 'CAVEGeodeShape' objects that have been selected */
    if (mGeodeShapeVector.size() > 0)
    {
        CAVEGeodeShapeVector::iterator itrGeodeShape;
        for (itrGeodeShape = mGeodeShapeVector.begin(); itrGeodeShape != mGeodeShapeVector.end(); itrGeodeShape++)
            (*itrGeodeShape)->mDOCollectorIndex = -1;
    }
    mMode = COLLECT_NONE;
    updateVertexMaskingVector();

    mGroupIconSurfaceVector.clear();
    mEditGeodeWireframeVector.clear();
    mDOIconSurfaceSwitch->removeChildren(0, mGeodeShapeVector.size());
    mDOGeodeWireframeSwitch->removeChildren(0, mGeodeShapeVector.size());
    mGeodeShapeVector.clear();
}


/***************************************************************
* Function: setAllIconSurfaceHighlights()
*
* These functions set highlights of all 'CAVEGeodeIconSurface'
* objects under 'mDOIconSurfaceSwitch' 
*
***************************************************************/
void DOGeometryCollector::setAllIconSurfaceHighlightsNormal()
{
    int nGroupIconSurface = mGroupIconSurfaceVector.size();
    if (nGroupIconSurface <= 0) return;
    for (int i = 0; i < nGroupIconSurface; i++)
    {
        mGroupIconSurfaceVector[i]->setHighlightNormal();
    }
}


void DOGeometryCollector::setAllIconSurfaceHighlightsUnselected()
{
    int nGroupIconSurface = mGroupIconSurfaceVector.size();
    if (nGroupIconSurface <= 0) return;
    for (int i = 0; i < nGroupIconSurface; i++)
    {
        mGroupIconSurfaceVector[i]->setHighlightUnselected();
    }
}


/***************************************************************
* Function: updateVertexMaskingVector()
***************************************************************/
void DOGeometryCollector::updateVertexMaskingVector()
{
    const int numGeodeShapes = mGeodeShapeVector.size();
    if (numGeodeShapes <= 0) return;

    /* update vertex masking vector for each 'CAVEGeodeShape' object */
    for (int i = 0; i < numGeodeShapes; i++)
    {
        CAVEGeodeShape *orgGeodePtr = mGeodeShapeVector[i];
        CAVEGeodeShape *refGeodePtr = dynamic_cast <CAVEGeodeShape*> (mDOShapeRefMatrixTrans->getChild(i));

        /* 'COLLECT_NONE': nothing has been selected, set all vertice inactive */
        if (mMode == COLLECT_NONE) orgGeodePtr->updateVertexMaskingVector(false);

        /* 'COLLECT_GEODE': no specific surface selected, set all vertices active */
        else if (mMode == COLLECT_GEODE) orgGeodePtr->updateVertexMaskingVector(true);

        /* 'COLLECT_GEOMETRY': some surfaces selected, set affiliated vertices active */
        else if (mMode == COLLECT_GEOMETRY) orgGeodePtr->updateVertexMaskingVector();

        /* synchronize the 'mVertexMaskingVector' field of actual 'CAVEGeode' and the cloned reference. */
        if (refGeodePtr) 
            refGeodePtr->updateVertexMaskingVector(orgGeodePtr->getVertexMaskingVector());
    }
}

