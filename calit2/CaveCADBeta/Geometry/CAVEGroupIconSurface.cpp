/***************************************************************
* File Name: CAVEGroupIconSurface.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Jan 26, 2011
*
***************************************************************/
#include "CAVEGroupIconSurface.h"


using namespace std;
using namespace osg;


/* surface resizing and translation hints */
Vec3 CAVEGroupIconSurface::gShapeCenter(Vec3(0, 0, 0));
Vec3 CAVEGroupIconSurface::gIconCenter(Vec3(0, 0, 0));
float CAVEGroupIconSurface::gScaleVal(1.0f);


// Constructor
CAVEGroupIconSurface::CAVEGroupIconSurface(): mPrimaryFlag(false)
{
    mRootTrans = new MatrixTransform();
    addChild(mRootTrans);

    mCAVEGeodeIconVector.clear();
}


/***************************************************************
* Function: acceptCAVEGeodeShape()
*
* Compared with original coordinates data in 'CAVEGeodeShape', 
* generated coordinates in 'CAVEGeodeIconSurface' is eaxctly the
* the same as those appeared in 'CAVEGeodeShape'. The scaling and
* translation to 'gIconCenter' effects are impletemented by its
* acendent 'PositionAltitudeTransform' object.
* 
***************************************************************/
void CAVEGroupIconSurface::acceptCAVEGeodeShape(CAVEGeodeShape *shapeGeode, CAVEGeodeShape *shapeGeodeRef)
{
    mCAVEGeodeShapeOriginPtr = shapeGeode;

    CAVEGeometryVector &orgGeomVector = shapeGeode->getCAVEGeometryVector();
    CAVEGeometryVector &refGeomVector = shapeGeodeRef->getCAVEGeometryVector();
    int nGeoms = orgGeomVector.size();
    if (nGeoms <= 0) return;

    /* re-generate vertex coordinate list, keep normal and texcoords the same from 'CAGEGeodeShape' */
    mSurfVertexArray = new Vec3Array;
    mSurfNormalArray = new Vec3Array;
    mSurfUDirArray = new Vec3Array;
    mSurfVDirArray = new Vec3Array;
    mSurfTexcoordArray = new Vec2Array;

    Vec3Array* geodeVertexArray = shapeGeode->mVertexArray;
    Vec3Array* geodeNormalArray = shapeGeode->mNormalArray;
    Vec3Array* geodeUDirArray = shapeGeode->mUDirArray;
    Vec3Array* geodeVDirArray = shapeGeode->mVDirArray;
    Vec2Array* geodeTexcoordArray = shapeGeode->mTexcoordArray;

    Vec3 *geodeVertexDataPtr, *geodeNormalDataPtr, *geodeUDirDataPtr, *geodeVDirDataPtr;
    Vec2 *geodeTexcoordDataPtr;

    /* check the valid status of all data field from 'CAVEGeodeShape' */
    if (geodeVertexArray->getType() == Array::Vec3ArrayType)
        geodeVertexDataPtr = (Vec3*) (geodeVertexArray->getDataPointer());
    else return;

    if (geodeNormalArray->getType() == Array::Vec3ArrayType)
        geodeNormalDataPtr = (Vec3*) (geodeNormalArray->getDataPointer());
    else return;

    if (geodeUDirArray->getType() == Array::Vec3ArrayType)
        geodeUDirDataPtr = (Vec3*) (geodeUDirArray->getDataPointer());
    else return;

    if (geodeVDirArray->getType() == Array::Vec3ArrayType)
        geodeVDirDataPtr = (Vec3*) (geodeVDirArray->getDataPointer());
    else return;

    if (geodeTexcoordArray->getType() == Array::Vec2ArrayType)
        geodeTexcoordDataPtr = (Vec2*) (geodeTexcoordArray->getDataPointer());
    else return;

    /* convert vertex coordinates from CAVEGeodeShape space to CAVEGeodeIcon space */
    int nVerts = shapeGeode->mNumVertices;
    for (int i = 0; i < nVerts; i++) 
    {
        mSurfVertexArray->push_back(geodeVertexDataPtr[i]);
    }

    /* preserve the same normals and texture coordinates */
    int nNormals = shapeGeode->mNumNormals;
    for (int i = 0; i < nNormals; i++)
    {
        mSurfNormalArray->push_back(geodeNormalDataPtr[i]);
        mSurfUDirArray->push_back(geodeUDirDataPtr[i]);
        mSurfVDirArray->push_back(geodeVDirDataPtr[i]);
    }

    int nTexcoords = shapeGeode->mNumTexcoords;
    for (int i = 0; i < nTexcoords; i++) 
    {
        mSurfTexcoordArray->push_back(geodeTexcoordDataPtr[i]);
    }

    /* apply offset from 'gShapeCenter' to Vec3(0, 0, 0) on root level */
    Matrixf transMat;
    transMat.makeTranslate(-gShapeCenter);
    mRootTrans->setMatrix(transMat);

    /* copy CAVEGeometry objects into separate 'CAVEGeodeIconSurface' */
    for (int i = 0; i < nGeoms; i++)
    {
        CAVEGeodeIconSurface *iconSurface = new CAVEGeodeIconSurface(&mSurfVertexArray, &mSurfNormalArray,
                        &mSurfTexcoordArray, &(orgGeomVector[i]), &(refGeomVector[i]));

        /* 1) take record of 'iconSurface' in mCAVEGeodeIconVector; 2) add it to 'this' group */
        mCAVEGeodeIconVector.push_back(iconSurface);
        mRootTrans->addChild(iconSurface);
    }
}


/***************************************************************
* Function: setHighlightNormal()
***************************************************************/
void CAVEGroupIconSurface::setHighlightNormal()
{
    int nIconSurface = mCAVEGeodeIconVector.size();
    if (nIconSurface <= 0) 
        return;
    for (int i = 0; i < nIconSurface; i++)
    {
        CAVEGeodeIconSurface *iconSurfPtr = dynamic_cast <CAVEGeodeIconSurface*> (mCAVEGeodeIconVector[i]);
        if (iconSurfPtr) 
        {
            iconSurfPtr->setHighlightNormal();
        }
    }
}


/***************************************************************
* Function: setHighlightNormal()
***************************************************************/
void CAVEGroupIconSurface::setHighlightSelected()
{
    int nIconSurface = mCAVEGeodeIconVector.size();
    if (nIconSurface <= 0) 
        return;
    for (int i = 0; i < nIconSurface; i++)
    {
        CAVEGeodeIconSurface *iconSurfPtr = dynamic_cast <CAVEGeodeIconSurface*> (mCAVEGeodeIconVector[i]);
        if (iconSurfPtr) 
        {
            iconSurfPtr->setHighlightSelected();
        }
    }
}


/***************************************************************
* Function: setHighlightUnselected()
***************************************************************/
void CAVEGroupIconSurface::setHighlightUnselected()
{
    int nIconSurface = mCAVEGeodeIconVector.size();
    if (nIconSurface <= 0) 
        return;
    for (int i = 0; i < nIconSurface; i++)
    {
        CAVEGeodeIconSurface *iconSurfPtr = dynamic_cast <CAVEGeodeIconSurface*> (mCAVEGeodeIconVector[i]);
        if (iconSurfPtr) 
        {
            iconSurfPtr->setHighlightUnselected();
        }
    }
}


/***************************************************************
* Function: applyEditingFinishes()
***************************************************************/
void CAVEGroupIconSurface::applyEditingFinishes(CAVEGeodeShape::EditorInfo **infoPtr)
{
    /* call generic static function to adapt 'EditorInfo' changes into array data */
    CAVEGeodeShape::applyEditorInfo(&mSurfVertexArray, &mSurfNormalArray, 
		&mSurfUDirArray, &mSurfVDirArray, &mSurfTexcoordArray,
		mSurfVertexArray, mSurfNormalArray, mSurfUDirArray, mSurfVDirArray, mSurfTexcoordArray,
		mSurfVertexArray->getNumElements(), infoPtr, mCAVEGeodeShapeOriginPtr->getVertexMaskingVector());

    const int nIconSurfs = mCAVEGeodeIconVector.size();
    if (nIconSurfs > 0)
    {
        for (int i = 0; i < nIconSurfs; i++)
        {
            CAVEGeodeIconSurface *geodeIconSurfacePtr = dynamic_cast <CAVEGeodeIconSurface*> (mCAVEGeodeIconVector[i]);
            if (geodeIconSurfacePtr)
            {
                geodeIconSurfacePtr->getGeometry()->dirtyDisplayList();
                geodeIconSurfacePtr->getGeometry()->dirtyBound();
            }
        }
    }
}


/***************************************************************
* Function: setSurfaceTranslationHint()
***************************************************************/
void CAVEGroupIconSurface::setSurfaceTranslationHint(const Vec3 &shapeCenter, const Vec3 &iconCenter)
{
    gShapeCenter = shapeCenter;
    gIconCenter = iconCenter;
}


/***************************************************************
* Function: getSurfaceTranslationHint()
***************************************************************/
void CAVEGroupIconSurface::getSurfaceTranslationHint(Vec3 &shapeCenter, Vec3 &iconCenter)
{
    shapeCenter = gShapeCenter;
    iconCenter = gIconCenter;
}


/***************************************************************
* Function: getSurfaceTranslationIconCenterHint()
***************************************************************/
void CAVEGroupIconSurface::getSurfaceTranslationIconCenterHint(osg::Vec3 &iconCenter)
{
    iconCenter = gIconCenter;
}


/***************************************************************
* Function: setSurfaceScalingHint()
***************************************************************/
void CAVEGroupIconSurface::setSurfaceScalingHint(const BoundingBox& boundingbox)
{
    gScaleVal = (CAVEGeodeIcon::gSphereBoundRadius) / (boundingbox.radius());
}


/***************************************************************
* Function: getSurfaceScalingHint()
***************************************************************/
const float &CAVEGroupIconSurface::getSurfaceScalingHint()
{
    return gScaleVal;
}

