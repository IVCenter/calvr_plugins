/***************************************************************
* File Name: CAVEGeodeSnapSolidshape.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Nov 18, 2010
*
***************************************************************/
#include "CAVEGeodeSnapSolidshape.h"


using namespace std;
using namespace osg;


//Constructor: CAVEGeodeSnapSolidshape
CAVEGeodeSnapSolidshape::CAVEGeodeSnapSolidshape(): mSnappingUnitDist(0.0f)
{
    mInitPosition = Vec3(0, 0, 0);
    mScaleVect = Vec3(1, 1, 1);

    Material* material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 0.5f, 0.0f, 1.0f));
    material->setSpecular(Material::FRONT_AND_BACK, osg::Vec4( 1.0f, 0.5f, 0.0f, 1.0f));
    material->setAlpha(Material::FRONT, 0.8f);

    StateSet* stateset = new StateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    stateset->setMode(GL_LIGHTING, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    setStateSet(stateset);
}


/***************************************************************
* Function: isValid()
***************************************************************/
bool CAVEGeodeSnapSolidshape::isValid()
{
    if (mScaleVect.length() >= mSnappingUnitDist) 
        return true;
    else 
        return false;
}


/***************************************************************
* Function: setInitPosition()
***************************************************************/
void CAVEGeodeSnapSolidshape::setInitPosition(const osg::Vec3 &initPos, bool snap)
{
    if (snap)
    {
        /* round intersected position to integer multiples of 'mSnappingUnitDist' */
        Vec3 initPosRounded;
        float snapUnitX, snapUnitY, snapUnitZ;
        snapUnitX = snapUnitY = snapUnitZ = mSnappingUnitDist;
        if (initPos.x() < 0) 
            snapUnitX = -mSnappingUnitDist;
        if (initPos.y() < 0) 
            snapUnitY = -mSnappingUnitDist;
        if (initPos.z() < 0) 
            snapUnitZ = -mSnappingUnitDist;

        int xSeg = (int)(abs((int)((initPos.x() + 0.5 * snapUnitX) / mSnappingUnitDist)));
        int ySeg = (int)(abs((int)((initPos.y() + 0.5 * snapUnitY) / mSnappingUnitDist)));
        int zSeg = (int)(abs((int)((initPos.z() + 0.5 * snapUnitZ) / mSnappingUnitDist)));

        initPosRounded.x() = xSeg * snapUnitX;
        initPosRounded.y() = ySeg * snapUnitY;
        initPosRounded.z() = zSeg * snapUnitZ;

        mInitPosition = initPosRounded;
    }
    else
    {
        mInitPosition = initPos;
    }
}


// Constructor: CAVEGeodeSnapSolidshapeBox
CAVEGeodeSnapSolidshapeBox::CAVEGeodeSnapSolidshapeBox()
{
    mBox = new Box(Vec3(0.5, 0.5, 0.5), 0.5);
    Drawable* boxDrawable = new ShapeDrawable(mBox);
    addDrawable(boxDrawable);
}


// Constructor: CAVEGeodeSnapSolidshapeCylinder
CAVEGeodeSnapSolidshapeCylinder::CAVEGeodeSnapSolidshapeCylinder()
{
    mCylinder = new Cylinder(Vec3(0, 0, 0), 1.0, 1.0);
    Drawable* cylinderDrawable = new ShapeDrawable(mCylinder);
    addDrawable(cylinderDrawable);
}

// Constructor: CAVEGeodeSnapSolidshapeCone
CAVEGeodeSnapSolidshapeCone::CAVEGeodeSnapSolidshapeCone()
{
    mCone = new Cone(Vec3(0, 0, 0), 1.0, 1.0);
    Drawable* coneDrawable = new ShapeDrawable(mCone);
    addDrawable(coneDrawable);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeSnapSolidshapeBox::resize(const osg::Vec3 &gridVect, bool snap)
{
    if (snap)
        mScaleVect = gridVect * mSnappingUnitDist;
    else
        mScaleVect = gridVect;

    mBox = new Box;
    
    //std::cout << snap << std::endl;
    //mBox->setCenter(mInitPosition + mScaleVect * 0.5);
    
    // Keep the initial position the same - snapping
    mBox->setCenter(mInitPosition + mScaleVect * 0.5);

    mBox->setHalfLengths(mScaleVect * 0.5);
    Drawable* boxDrawable = new ShapeDrawable(mBox);
    setDrawable(0, boxDrawable);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeSnapSolidshapeCylinder::resize(const osg::Vec3 &gridVect, bool snap)
{
    mScaleVect = gridVect * mSnappingUnitDist;
    float rad = sqrt(mScaleVect.x() * mScaleVect.x() + mScaleVect.y() * mScaleVect.y());
    mCylinder = new Cylinder;

    mCylinder->setCenter(mInitPosition + Vec3(0, 0, mScaleVect.z() * 0.5));
    mCylinder->setRadius(rad);
    mCylinder->setHeight(mScaleVect.z());
    Drawable* cylinderDrawable = new ShapeDrawable(mCylinder);
    setDrawable(0, cylinderDrawable);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeSnapSolidshapeCone::resize(const osg::Vec3 &gridVect, bool snap)
{
    mScaleVect = gridVect * mSnappingUnitDist;
    float rad = sqrt(mScaleVect.x() * mScaleVect.x() + mScaleVect.y() * mScaleVect.y());
    mCone = new Cone;

    mCone->setCenter(mInitPosition + Vec3(0, 0, mScaleVect.z() * 0.5));
    mCone->setRadius(rad);
    mCone->setHeight(mScaleVect.z());
    Drawable* coneDrawable = new ShapeDrawable(mCone);
    setDrawable(0, coneDrawable);
}
