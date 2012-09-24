/***************************************************************
* File Name: CAVEGeodeReference.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Nov 30, 2010
*
***************************************************************/
#include "CAVEGeodeReference.h"


using namespace std;
using namespace osg;


const int CAVEGeodeReferenceAxis::gHeadUnitSegments(2);
const float CAVEGeodeReferenceAxis::gBodyRadius(0.003f);
const float CAVEGeodeReferenceAxis::gArrowRadius(0.005f);
const float CAVEGeodeReferenceAxis::gArrowLength(0.05f);
const float CAVEGeodeReferencePlane::gSideLength(20.f);

// Constructor
CAVEGeodeReferenceAxis::CAVEGeodeReferenceAxis(): mType(POS_Z)
{
    float len = (CAVEGeodeSnapWireframe::gSnappingUnitDist) * gHeadUnitSegments;

    mCone = new Cone();
    mConeDrawable = new ShapeDrawable(mCone);
    mCone->setRadius(gArrowRadius);
    mCone->setHeight(gArrowLength);
    mCone->setCenter(Vec3(0, 0, len));
    addDrawable(mConeDrawable);

    mCylinder = new Cylinder();
    mCylinderDrawable = new ShapeDrawable(mCylinder);
    mCylinder->setRadius(gBodyRadius);
    mCylinder->setHeight(len);
    mCylinder->setCenter(Vec3(0, 0, len * 0.5));
    addDrawable(mCylinderDrawable);

    /* texture coordinates is associated with size of the geometry */
    Material *material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0, 1, 0, 1));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0f);

    StateSet *stateset = getOrCreateStateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
}


/***************************************************************
* Function: setType()
***************************************************************/
void CAVEGeodeReferenceAxis::setType(const AxisType &typ, osg::MatrixTransform **matTrans)
{
    if (typ == mType) 
        return;

    mType = typ;

    Quat rotation = Quat(0.f, Vec3(0, 0, 1));
    switch (mType)
    {
        case POS_X: rotation = Quat(M_PI * 0.5, Vec3(0, 1, 0));  break;
        case POS_Y: rotation = Quat(M_PI * 0.5, Vec3(-1, 0, 0)); break;
        case POS_Z: rotation = Quat(0.f, Vec3(0, 0, 1));         break;
        case NEG_X: rotation = Quat(M_PI * 0.5, Vec3(0, -1, 0)); break;
        case NEG_Y: rotation = Quat(M_PI * 0.5, Vec3(1, 0, 0));	 break;
        case NEG_Z: rotation = Quat(M_PI, Vec3(0, 0, 1));        break;
        default: break;
    }

    Matrixf rotMat;
    rotMat.setRotate(rotation);
    (*matTrans)->setMatrix(rotMat);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeReferenceAxis::resize(const float &length)
{
    float len = length > 0 ? length: -length;
    len += (CAVEGeodeSnapWireframe::gSnappingUnitDist) * gHeadUnitSegments;

    mCone->setRadius(gArrowRadius);
    mCone->setHeight(gArrowLength);
    mCone->setCenter(Vec3(0, 0, len));

    mCylinder->setRadius(gBodyRadius);
    mCylinder->setHeight(len);
    mCylinder->setCenter(Vec3(0, 0, len * 0.5));

    mConeDrawable = new ShapeDrawable(mCone);
    mCylinderDrawable = new ShapeDrawable(mCylinder);
    setDrawable(0, mConeDrawable);
    setDrawable(1, mCylinderDrawable);
}


// Constructor
CAVEGeodeReferencePlane::CAVEGeodeReferencePlane()
{
    mVertexArray = new Vec3Array;
    mNormalArray = new Vec3Array;
    mTexcoordArray = new Vec2Array;
    mGeometry = new Geometry;

    /* vertex coordinates and normals */
    mVertexArray->push_back(Vec3(-gSideLength, -gSideLength, 0));	mNormalArray->push_back(Vec3(0, 0, 1));
    mVertexArray->push_back(Vec3( gSideLength, -gSideLength, 0));	mNormalArray->push_back(Vec3(0, 0, 1));
    mVertexArray->push_back(Vec3( gSideLength,  gSideLength, 0));	mNormalArray->push_back(Vec3(0, 0, 1));
    mVertexArray->push_back(Vec3(-gSideLength,  gSideLength, 0));	mNormalArray->push_back(Vec3(0, 0, 1));

    /* texture coordinates */
    float unitLength = SnapLevelController::getInitSnappingLength();
    float texcoordMax = gSideLength / unitLength;

    mTexcoordArray->push_back(Vec2(-texcoordMax, -texcoordMax));
    mTexcoordArray->push_back(Vec2( texcoordMax, -texcoordMax));
    mTexcoordArray->push_back(Vec2( texcoordMax,  texcoordMax));
    mTexcoordArray->push_back(Vec2(-texcoordMax,  texcoordMax));

    DrawElementsUInt* xyPlaneUp = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);  
    xyPlaneUp->push_back(0);  
    xyPlaneUp->push_back(1);    
    xyPlaneUp->push_back(2);    
    xyPlaneUp->push_back(3);
    xyPlaneUp->push_back(0);

    mGeometry->addPrimitiveSet(xyPlaneUp);
    mGeometry->setVertexArray(mVertexArray);
    mGeometry->setNormalArray(mNormalArray);
    mGeometry->setTexCoordArray(0, mTexcoordArray);
    mGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);
    addDrawable(mGeometry);

    /* texture coordinates is associated with size of the geometry */
    Material *material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    material->setAlpha(Material::FRONT_AND_BACK, 1.f);

    StateSet *stateset = getOrCreateStateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeReferencePlane::resize(const float lx, const float ly, const float unitsize)
{
    if (mVertexArray->getType() == Array::Vec3ArrayType)
    {
        Vec3* vertexArrayDataPtr = (Vec3*) (mVertexArray->getDataPointer());
        vertexArrayDataPtr[0] = Vec3(-lx, -ly, 0);
        vertexArrayDataPtr[1] = Vec3( lx, -ly, 0);
        vertexArrayDataPtr[2] = Vec3( lx,  ly, 0);
        vertexArrayDataPtr[3] = Vec3(-lx,  ly, 0);
    }

    if (mTexcoordArray->getType() == Array::Vec2ArrayType)
    {
        Vec2* texcoordArrayDataPtr = (Vec2*) (mTexcoordArray->getDataPointer());
        texcoordArrayDataPtr[0] = Vec2(-lx, -ly) / unitsize;
        texcoordArrayDataPtr[1] = Vec2( lx, -ly) / unitsize;
        texcoordArrayDataPtr[2] = Vec2( lx,  ly) / unitsize;
        texcoordArrayDataPtr[3] = Vec2(-lx,  ly) / unitsize;
    }

    mGeometry->dirtyDisplayList();
    mGeometry->dirtyBound();
}


/***************************************************************
* Function: setGridColor()
***************************************************************/
void CAVEGeodeReferencePlane::setGridColor(const GridColor &color)
{
    StateSet *stateset = getOrCreateStateSet();
    if (!stateset) 
        return;

    Image* texImage;
    if (color == RED) 
        texImage = osgDB::readImageFile(CAVEGeode::getDataDir() + "Textures/RedTile.PNG");
    else if (color == GREEN) 
        texImage = osgDB::readImageFile(CAVEGeode::getDataDir() + "Textures/GreenTile.PNG");
    else if (color == BLUE) 
        texImage = osgDB::readImageFile(CAVEGeode::getDataDir() + "Textures/BlueTile.PNG");
    else    
        return;

    Texture2D *texture = new Texture2D(texImage);
    texture->setWrap(Texture::WRAP_S, Texture::REPEAT);
    texture->setWrap(Texture::WRAP_T, Texture::REPEAT);
    texture->setDataVariance(Object::DYNAMIC);

    stateset->setTextureAttributeAndModes(0, texture, StateAttribute::ON);
}


/***************************************************************
* Function: setAlpha()
***************************************************************/
void CAVEGeodeReferencePlane::setAlpha(const float &alpha)
{
    StateSet *stateset = getOrCreateStateSet();
    if (!stateset) 
        return;

    Material *material = dynamic_cast<Material*> (stateset->getAttribute(StateAttribute::MATERIAL));
    if (!material) 
        return;

    material->setAlpha(Material::FRONT_AND_BACK, alpha);
    stateset->setAttributeAndModes(material, StateAttribute::ON);
}






















