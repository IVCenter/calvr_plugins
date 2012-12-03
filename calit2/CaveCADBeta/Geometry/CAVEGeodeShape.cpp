/***************************************************************
* File Name: CAVEGeodeShape.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Nov 23, 2010
*
***************************************************************/
#include "CAVEGeodeShape.h"


using namespace std;
using namespace osg;


const float CAVEGeodeShape::gTextureTileSize(0.3048f);
//const float CAVEGeodeShape::gTextureTileSize(304.8f);

// Constructor
CAVEGeodeShape::CAVEGeodeShape(const Type &typ, const Vec3 &initVect, const Vec3 &sVect):
		mCenterVect(Vec3(0, 0, 0)), mNumVertices(0), mNumNormals(0), mNumTexcoords(0),
		mDOCollectorIndex(-1)
{
    mVertexArray = new Vec3Array;
    mNormalArray = new Vec3Array;
    mUDirArray = new Vec3Array;
    mVDirArray = new Vec3Array;
    mTexcoordArray = new Vec2Array;
    mVertexMaskingVector.clear();

    switch (typ)
    {
        case BOX: initGeometryBox(initVect, sVect); break;
        case CYLINDER: initGeometryCylinder(initVect, sVect); break;
        default: break;
    }

    // texture coordinates is associated with size of the geometry
    Image* img = osgDB::readImageFile(CAVEGeode::getDataDir() + "Textures/White.JPG");
    Texture2D* texture = new Texture2D(img);
    texture->setWrap(Texture::WRAP_S, Texture::MIRROR);
    texture->setWrap(Texture::WRAP_T, Texture::MIRROR);

    Material *material = new Material;
    material->setSpecular(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0f);

    StateSet *stateset = getOrCreateStateSet();
    stateset->setTextureAttributeAndModes(0, texture, StateAttribute::ON);
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
}


/***************************************************************
* Constructor: CAVEGeodeShape()
*
* 'mDOCollectorIndex' will not be copied unless the shape is
*  selected by 'DOGeometryCollecotr'
*
***************************************************************/
CAVEGeodeShape::CAVEGeodeShape(CAVEGeodeShape *geodeShapeRef): mDOCollectorIndex(-1)
{
    mVertexArray = new Vec3Array;
    mNormalArray = new Vec3Array;
    mUDirArray = new Vec3Array;
    mVDirArray = new Vec3Array;
    mTexcoordArray = new Vec2Array;

    Vec3Array* geodeVertexArray = geodeShapeRef->mVertexArray;
    Vec3Array* geodeNormalArray = geodeShapeRef->mNormalArray;
    Vec3Array* geodeUDirArray = geodeShapeRef->mUDirArray;
    Vec3Array* geodeVDirArray = geodeShapeRef->mVDirArray;
    Vec2Array* geodeTexcoordArray = geodeShapeRef->mTexcoordArray;

    Vec3 *geodeVertexDataPtr, *geodeNormalDataPtr, *geodeUDirDataPtr, *geodeVDirDataPtr;
    Vec2 *geodeTexcoordDataPtr;

    // check the valid status of all data field from 'CAVEGeodeShape'
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

    mNumVertices = geodeShapeRef->mNumVertices;
    for (int i = 0; i < mNumVertices; i++) 
    {
        mVertexArray->push_back(geodeVertexDataPtr[i]);
    }

    mNumNormals = geodeShapeRef->mNumNormals;
    for (int i = 0; i < mNumNormals; i++)
    {
        mNormalArray->push_back(geodeNormalDataPtr[i]);
        mUDirArray->push_back(geodeUDirDataPtr[i]);
        mVDirArray->push_back(geodeVDirDataPtr[i]);
    }

    mNumTexcoords = geodeShapeRef->mNumTexcoords;
    for (int i = 0; i < mNumTexcoords; i++) 
    {
        mTexcoordArray->push_back(geodeTexcoordDataPtr[i]);
    }

    // make deep copy of 'CAVEGeometry' objects into this 'CAVEGeodeShape'
    CAVEGeometryVector &geomVector = geodeShapeRef->getCAVEGeometryVector();
    const int nGeoms = geomVector.size();
    if (nGeoms > 0)
    {
        for (int i = 0; i < nGeoms; i++)
        {
            CAVEGeometry *geometry = new CAVEGeometry(geomVector[i]);
            geometry->setVertexArray(mVertexArray);
            geometry->setNormalArray(mNormalArray);
            geometry->setTexCoordArray(0, mTexcoordArray);
            geometry->setNormalBinding(Geometry::BIND_PER_VERTEX);

            mGeometryVector.push_back(geometry);
            addDrawable(geometry);
        }
    }


    // copy the same center vector from 'geodeShapeRef', vertex masking vector is with all false values
    mCenterVect = geodeShapeRef->mCenterVect;
    mVertexMaskingVector.resize(mNumVertices, false);

    // apply color texture to virtual surface
    //applyColorTexture(geodeShapeRef->mDiffuse, geodeShapeRef->mSpecular, geodeShapeRef->mAlpha,
    //   geodeShapeRef->mTexFilename);
    applyColor(geodeShapeRef->mDiffuse, geodeShapeRef->mSpecular, geodeShapeRef->mAlpha);
    applyTexture(geodeShapeRef->mTexFilename);

}


// Destructor
CAVEGeodeShape::~CAVEGeodeShape()
{
}


/***************************************************************
* Function: updateVertexMaskingVector()
***************************************************************/
void CAVEGeodeShape::updateVertexMaskingVector(bool flag)
{
    mVertexMaskingVector.clear();
    mVertexMaskingVector.resize(mNumVertices, flag);
}


/***************************************************************
* Function: updateVertexMaskingVector()
***************************************************************/
void CAVEGeodeShape::updateVertexMaskingVector(const VertexMaskingVector &vertMaskingVector)
{
    updateVertexMaskingVector(false);

    if (vertMaskingVector.size() != mNumVertices)
    {
        cerr << "Warning: CAVEGeodeShape::updateVertexMaskingVector() could not apply vertex masking." << endl;
        return;
    }
    for (int i = 0; i < mNumVertices; i++) mVertexMaskingVector[i] = vertMaskingVector[i];
}


/***************************************************************
* Function: updateVertexMaskingVector()
*
* 'updateVertexMaskingVector' without input value: checks all
* 'CAVEGeometry' objects and mark vertices of selected geometries
*  in 'mVertexMaskingVector', and vertices clusters of them.
*
***************************************************************/
void CAVEGeodeShape::updateVertexMaskingVector()
{
    mVertexMaskingVector.clear();
    mVertexMaskingVector.resize(mNumVertices, false);

    if (mGeometryVector.size() <= 0) 
        return;

    for (int i = 0; i < mGeometryVector.size(); i++)
    {
        if (mGeometryVector[i]->mDOCollectorIndex >= 0)
        {
            // mark single indices contained in mGeometryVector[i]
            unsigned int nPrimitiveSets = mGeometryVector[i]->getNumPrimitiveSets();
            if (nPrimitiveSets > 0)
            {
                for (int j = 0; j < nPrimitiveSets; j++)
                {
                    PrimitiveSet *primSetRef = mGeometryVector[i]->getPrimitiveSet(j);

                    // support primitive set 'DrawElementsUInt', add more types of primitive sets here if needed
                    DrawElementsUInt* drawElementUIntRef = dynamic_cast <DrawElementsUInt*> (primSetRef);
                    if (drawElementUIntRef)
                    {
                        unsigned int nIdices = drawElementUIntRef->getNumIndices();
                        if (nIdices > 0)
                        {
                            for (int k = 0; k < nIdices; k++)
                            {
                                const int index = drawElementUIntRef->index(k);
                                mVertexMaskingVector[index] = true;
                            }
                        }
                    }
                }
            }

            // mark clustered indices contained in 'mIndexClusterVector'
            const int numClusters = mGeometryVector[i]->mIndexClusterVector.size();
            for (int j = 0; j < numClusters; j++)
            {
                CAVEGeometry::IndexClusterBase *clusterPtr = mGeometryVector[i]->mIndexClusterVector[j];
                for (int k = 0; k < clusterPtr->mNumIndices; k++)
                {
                    const int index = clusterPtr->mIndexVector[k];
                    mVertexMaskingVector[index] = true;
                }
            }
        }
    }
}


/***************************************************************
* Function: applyEditorInfo()
***************************************************************/
void CAVEGeodeShape::applyEditorInfo(EditorInfo **infoPtr)
{
    applyEditorInfo(infoPtr, this);
}


/***************************************************************
* Function: applyEditorInfo()
***************************************************************/
void CAVEGeodeShape::applyEditorInfo(EditorInfo **infoPtr, CAVEGeodeShape *refGeodePtr)
{
    // call generic static function to adapt 'EditorInfo' changes into array data
    applyEditorInfo(&mVertexArray, &mNormalArray, &mUDirArray, &mVDirArray, &mTexcoordArray,
		    	refGeodePtr->mVertexArray, refGeodePtr->mNormalArray,
			refGeodePtr->mUDirArray, refGeodePtr->mVDirArray, refGeodePtr->mTexcoordArray, 
		    mNumVertices, infoPtr, mVertexMaskingVector);

    // dirty display list and bound for all geometries
    const int nGeoms = mGeometryVector.size();
    if (nGeoms > 0)
    {
        for (int i = 0; i < nGeoms; i++)
        {
            mGeometryVector[i]->dirtyDisplayList();
            mGeometryVector[i]->dirtyBound();
        }
    }

    // update geode shape center
    const BoundingBox& bb = getBoundingBox();
    mCenterVect = bb.center();
}


/***************************************************************
* Function: hideSnapBounds()
***************************************************************/
void CAVEGeodeShape::hideSnapBounds()
{
    osg::Vec4 snapSphereColor = osg::Vec4(1, 0, 1, 0);
    Material *mat = new Material;
    mat->setDiffuse(Material::FRONT_AND_BACK, snapSphereColor);

    // hide all bounding spheres
    for (int i = 0 ; i < mVertBoundingSpheres.size(); ++i)
    {
        osg::ShapeDrawable * sd = mShapeDrawableMap[mVertBoundingSpheres[i]];

        osg::StateSet * ss = sd->getOrCreateStateSet();
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
    }
    
    // hide all bounding cylinders
    std::map<osg::Cylinder*, osg::ShapeDrawable*>::iterator it;

    for (it = mEdgeDrawableMap.begin(); it != mEdgeDrawableMap.end(); ++it)
    {
        osg::ShapeDrawable *sd = it->second;
        osg::StateSet *ss = sd->getOrCreateStateSet();
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
    }


}


/***************************************************************
* Function: snapToVertex()
***************************************************************/
bool CAVEGeodeShape::snapToVertex(const osg::Vec3 point, osg::Vec3 *ctr)
{
    osg::Vec4 hideColor, showColor;
    hideColor = osg::Vec4(1, 0, 1, 0);
    showColor = osg::Vec4(1, 0, 1, 0.6);
    Material *mat = new Material;
    mat->setDiffuse(Material::FRONT_AND_BACK, hideColor);

    // hide all vertex bounding spheres
    for (int i = 0 ; i < mVertBoundingSpheres.size(); ++i)
    {
        osg::ShapeDrawable * sd = mShapeDrawableMap[mVertBoundingSpheres[i]];

        osg::StateSet * ss = sd->getOrCreateStateSet();
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
    }

    std::map<osg::Cylinder*, osg::ShapeDrawable*>::iterator it;

    // hide all edge bounding cylinders
    for (it = mEdgeDrawableMap.begin(); it != mEdgeDrawableMap.end(); ++it)
    {
        osg::ShapeDrawable *sd = it->second;
        osg::StateSet *ss = sd->getOrCreateStateSet();
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
    }
   
    // check vertex intersection
    for (int i = 0 ; i < mVertBoundingSpheres.size(); ++i)
    {
        float distance, radius = 0.2;
        osg::Sphere * sph = mVertBoundingSpheres[i];
        osg::Vec3 center = (osg::Vec3)sph->getCenter();
        distance = (point[0] - center[0]) * (point[0] - center[0]) +
                   (point[1] - center[1]) * (point[1] - center[1]) +
                   (point[2] - center[2]) * (point[2] - center[2]);
        distance = pow(distance, 0.5);
    
        // if pointer position inside sphere
        if (distance <= 0.2)
        {
            *ctr = center;

            mat = new Material;
            mat->setDiffuse(Material::FRONT_AND_BACK, showColor);
            osg::StateSet * ss = mShapeDrawableMap[sph]->getOrCreateStateSet();
            ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);

            return true;
        }
    }
    
    // check edge intersection
    for (it = mEdgeDrawableMap.begin(); it != mEdgeDrawableMap.end(); ++it)
    {
        osg::Cylinder *cyl = it->first;
        osg::Geode *node = mEdgeGeodeMap[cyl];

        if (node->getBoundingBox().contains(point))
        {
            mat = new Material;
            mat->setDiffuse(Material::FRONT_AND_BACK, showColor);
            osg::StateSet * ss = it->second->getOrCreateStateSet();
            ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
            
            osg::Vec3 center = cyl->getCenter();

            // aligned on x axis
            if (cyl->getRotation() == osg::Quat(M_PI/2, osg::Vec3(0, 1, 0)))
            {
                *ctr = osg::Vec3(point[0], center[1], center[2]);
            }

            // aligned on y axis
            else if (cyl->getRotation() == osg::Quat(M_PI/2, osg::Vec3(1, 0, 0)))
            {
                *ctr = osg::Vec3(center[0], point[1], center[2]);
            }

            // aligned on y axis
            else
            {
                *ctr = osg::Vec3(center[0], center[1], point[2]);
            }
            return true;
        }
    }
    return false;
}


/***************************************************************
* Function: initGeometryBox()
***************************************************************/
void CAVEGeodeShape::initGeometryBox(const Vec3 &initVect, const Vec3 &sVect)
{
    float xMin, yMin, zMin, xMax, yMax, zMax;
    xMin = initVect.x();	xMax = initVect.x() + sVect.x();
    yMin = initVect.y();	yMax = initVect.y() + sVect.y();
    zMin = initVect.z();	zMax = initVect.z() + sVect.z();
    if (xMin > xMax) 
    { 
        xMin = initVect.x() + sVect.x();	
        xMax = initVect.x(); 
    }
    if (yMin > yMax) 
    { 
        yMin = initVect.y() + sVect.y();	
        yMax = initVect.y(); 
    }
    if (zMin > zMax) 
    { 
        zMin = initVect.z() + sVect.z();	
        zMax = initVect.z();
    }

    Vec3 up, down, front, back, left, right;
    up = Vec3(0, 0, 1);		down = Vec3(0, 0, -1);
    front = Vec3(0, -1, 0);	back = Vec3(0, 1, 0);
    left = Vec3(-1, 0, 0);	right = Vec3(1, 0, 0);

    // decide x, y, z span and write 'mCenterVect'
    float xspan = xMax - xMin, yspan = yMax - yMin, zspan = zMax - zMin;
    mCenterVect = Vec3((xMax + xMin) * 0.5, (yMax + yMin) * 0.5, (zMax + zMin) * 0.5);
    mNumVertices = mNumNormals = mNumTexcoords = 24;

    mVertexArray->push_back(Vec3(xMax, yMax, zMax));	mNormalArray->push_back(up);
    mVertexArray->push_back(Vec3(xMin, yMax, zMax));	mNormalArray->push_back(up);
    mVertexArray->push_back(Vec3(xMin, yMin, zMax));	mNormalArray->push_back(up);
    mVertexArray->push_back(Vec3(xMax, yMin, zMax));	mNormalArray->push_back(up);
    mVertexArray->push_back(Vec3(xMax, yMax, zMin));	mNormalArray->push_back(down);
    mVertexArray->push_back(Vec3(xMin, yMax, zMin));	mNormalArray->push_back(down);
    mVertexArray->push_back(Vec3(xMin, yMin, zMin));	mNormalArray->push_back(down);
    mVertexArray->push_back(Vec3(xMax, yMin, zMin));	mNormalArray->push_back(down);
    
    // Bounding cylinders for edges

    float radius = 0.2;
    float snapSphereRadius = 0.2, snapSphereOpacity = 0.5;
    osg::Vec4 snapSphereColor = osg::Vec4(1, 0, 1, 0); 

    osg::Vec3 center, ulfront, urfront, llfront, lrfront, ulback, urback, llback, lrback;
    osg::Cylinder *cyl;
    osg::ShapeDrawable *shpDrawable;
    osg::PositionAttitudeTransform *pat;
    osg::Geode *geode;

    Material *mat;
    mat = new osg::Material();
    mat->setDiffuse(Material::FRONT_AND_BACK, snapSphereColor);


    ulfront = osg::Vec3(xMin, yMin, zMax);
    urfront = osg::Vec3(xMax, yMin, zMax);
    llfront = osg::Vec3(xMin, yMin, zMin);
    lrfront = osg::Vec3(xMax, yMin, zMin);

    ulback = osg::Vec3(xMin, yMax, zMax);
    urback = osg::Vec3(xMax, yMax, zMax);
    llback = osg::Vec3(xMin, yMax, zMin);
    lrback = osg::Vec3(xMax, yMax, zMin);
    
    std::vector<osg::Cylinder*> cylVec;
    // top
    center = ulfront + osg::Vec3(xspan/2, 0, 0);
    cyl = new osg::Cylinder(center, radius, xspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(0, 1, 0)));
    cylVec.push_back(cyl);

    center = urfront + osg::Vec3(0, yspan/2, 0);
    cyl = new osg::Cylinder(center, radius, yspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(1, 0, 0)));
    cylVec.push_back(cyl);

    center = urback + osg::Vec3(-xspan/2, 0, 0);
    cyl = new osg::Cylinder(center, radius, xspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(0, 1, 0)));
    cylVec.push_back(cyl);

    center = ulback + osg::Vec3(0, -yspan/2, 0);
    cyl = new osg::Cylinder(center, radius, yspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(1, 0, 0)));
    cylVec.push_back(cyl);

    // sides
    center = urfront + osg::Vec3(0, 0, -zspan/2);
    cyl = new osg::Cylinder(center, radius, zspan - radius*2);
    cylVec.push_back(cyl);

    center = urback + osg::Vec3(0, 0, -zspan/2);
    cyl = new osg::Cylinder(center, radius, zspan - radius*2);
    cylVec.push_back(cyl);

    center = ulback + osg::Vec3(0, 0, -zspan/2);
    cyl = new osg::Cylinder(center, radius, zspan - radius*2);
    cylVec.push_back(cyl);

    center = ulfront + osg::Vec3(0, 0, -zspan/2);
    cyl = new osg::Cylinder(center, radius, zspan - radius*2);
    cylVec.push_back(cyl);

    // bottom
    center = llfront + osg::Vec3(xspan/2, 0, 0);
    cyl = new osg::Cylinder(center, radius, xspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(0, 1, 0)));
    cylVec.push_back(cyl);

    center = lrfront + osg::Vec3(0, yspan/2, 0);
    cyl = new osg::Cylinder(center, radius, yspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(1, 0, 0)));
    cylVec.push_back(cyl);

    center = lrback + osg::Vec3(-xspan/2, 0, 0);
    cyl = new osg::Cylinder(center, radius, xspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(0, 1, 0)));
    cylVec.push_back(cyl);

    center = llback + osg::Vec3(0, -yspan/2, 0);
    cyl = new osg::Cylinder(center, radius, yspan - radius*2);
    cyl->setRotation(osg::Quat(M_PI/2, osg::Vec3(1, 0, 0)));
    cylVec.push_back(cyl);

    osg::StateSet *ss;
    for (int i = 0; i < cylVec.size(); ++i)
    {
        shpDrawable = new osg::ShapeDrawable(cylVec[i]);
        ss = shpDrawable->getOrCreateStateSet();
        ss->setMode(GL_BLEND, StateAttribute::PROTECTED | StateAttribute::ON );
        ss->setRenderingHint(StateAttribute::PROTECTED | StateSet::TRANSPARENT_BIN);
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
        ss->setMode(GL_CULL_FACE, StateAttribute::PROTECTED| StateAttribute::ON);

        addDrawable(shpDrawable);

        mEdgeDrawableMap[cylVec[i]] = shpDrawable; 
        geode = new osg::Geode();
        geode->addDrawable(shpDrawable);
        mEdgeGeodeMap[cylVec[i]] = geode;
    }


    // Add vertex bounding spheres
    for (int i = 0; i < 8; ++i)
    {
        osg::Sphere *sph =  new osg::Sphere(mVertexArray->at(i), snapSphereRadius);
        osg::ShapeDrawable *shpDraw = new osg::ShapeDrawable(sph);

        Material *mat = new Material;
        mat->setDiffuse(Material::FRONT_AND_BACK, snapSphereColor);

        StateSet *ss = shpDraw->getOrCreateStateSet();
        ss->setMode(GL_BLEND, StateAttribute::PROTECTED | StateAttribute::ON );
        ss->setRenderingHint(StateAttribute::PROTECTED | StateSet::TRANSPARENT_BIN);
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
        ss->setMode(GL_CULL_FACE, StateAttribute::PROTECTED| StateAttribute::ON);

        addDrawable(shpDraw);
        mVertBoundingSpheres.push_back(sph);
        mShapeDrawableMap[sph] = shpDraw;
    }

    for (int i = 0; i < 8; i++) 
    { 
        mUDirArray->push_back(right);	
        mVDirArray->push_back(back); 
    }
    for (int i = 0; i < 2; i++)
    {
        mTexcoordArray->push_back(Vec2(xspan, yspan) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(0, yspan) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(0, 0) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(xspan, 0) / gTextureTileSize);
    }

    mVertexArray->push_back(Vec3(xMax, yMin, zMax));	mNormalArray->push_back(front);
    mVertexArray->push_back(Vec3(xMin, yMin, zMax));	mNormalArray->push_back(front);
    mVertexArray->push_back(Vec3(xMin, yMin, zMin));	mNormalArray->push_back(front);
    mVertexArray->push_back(Vec3(xMax, yMin, zMin)); 	mNormalArray->push_back(front);
    mVertexArray->push_back(Vec3(xMax, yMax, zMax));	mNormalArray->push_back(back);
    mVertexArray->push_back(Vec3(xMin, yMax, zMax));	mNormalArray->push_back(back);
    mVertexArray->push_back(Vec3(xMin, yMax, zMin));	mNormalArray->push_back(back);
    mVertexArray->push_back(Vec3(xMax, yMax, zMin));	mNormalArray->push_back(back);

    for (int i = 0; i < 8; i++) 
    {
        mUDirArray->push_back(right);	
        mVDirArray->push_back(up); 
    }
    for (int i = 0; i < 2; i++)
    {
        mTexcoordArray->push_back(Vec2(xspan, zspan) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(0, zspan) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(0, 0) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(xspan, 0) / gTextureTileSize);
    }

    mVertexArray->push_back(Vec3(xMin, yMax, zMax));	mNormalArray->push_back(left);
    mVertexArray->push_back(Vec3(xMin, yMin, zMax));	mNormalArray->push_back(left);
    mVertexArray->push_back(Vec3(xMin, yMin, zMin));	mNormalArray->push_back(left);
    mVertexArray->push_back(Vec3(xMin, yMax, zMin));	mNormalArray->push_back(left);
    mVertexArray->push_back(Vec3(xMax, yMax, zMax));	mNormalArray->push_back(right);
    mVertexArray->push_back(Vec3(xMax, yMin, zMax));	mNormalArray->push_back(right);
    mVertexArray->push_back(Vec3(xMax, yMin, zMin));	mNormalArray->push_back(right);
    mVertexArray->push_back(Vec3(xMax, yMax, zMin)); 	mNormalArray->push_back(right);

    for (int i = 0; i < 8; i++) 
    { 
        mUDirArray->push_back(back);	
        mVDirArray->push_back(up); 
    }
    for (int i = 0; i < 2; i++)
    {
        mTexcoordArray->push_back(Vec2(yspan, zspan) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(0, zspan) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(0, 0) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(yspan, 0) / gTextureTileSize);
    }

    // create geometries for each surface
    CAVEGeometry **geometryArrayPtr = new CAVEGeometry*[6];
    for (int i = 0; i < 6; i++)
    {
        geometryArrayPtr[i] = new CAVEGeometry;
        geometryArrayPtr[i]->setVertexArray(mVertexArray);
        geometryArrayPtr[i]->setNormalArray(mNormalArray);
        geometryArrayPtr[i]->setTexCoordArray(0, mTexcoordArray);
        geometryArrayPtr[i]->setNormalBinding(Geometry::BIND_PER_VERTEX);

        mGeometryVector.push_back(geometryArrayPtr[i]);
        addDrawable(geometryArrayPtr[i]);
    }

    // write primitive set and index clusters
    DrawElementsUInt* topSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);  
    DrawElementsUInt* bottomSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);
    DrawElementsUInt* frontSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);  
    DrawElementsUInt* backSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);
    DrawElementsUInt* leftSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);  
    DrawElementsUInt* rightSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);

    topSurface->push_back(0);    	bottomSurface->push_back(4);
    topSurface->push_back(1);    	bottomSurface->push_back(7);
    topSurface->push_back(2);    	bottomSurface->push_back(6);
    topSurface->push_back(3);    	bottomSurface->push_back(5);
    topSurface->push_back(0);    	bottomSurface->push_back(4);

    frontSurface->push_back(8);		backSurface->push_back(12);
    frontSurface->push_back(9);		backSurface->push_back(15);
    frontSurface->push_back(10);	backSurface->push_back(14);
    frontSurface->push_back(11);	backSurface->push_back(13);
    frontSurface->push_back(8);		backSurface->push_back(12);

    leftSurface->push_back(16);		rightSurface->push_back(20);
    leftSurface->push_back(19);		rightSurface->push_back(21);
    leftSurface->push_back(18);		rightSurface->push_back(22);
    leftSurface->push_back(17);		rightSurface->push_back(23);
    leftSurface->push_back(16);		rightSurface->push_back(20);

    geometryArrayPtr[0]->addPrimitiveSet(topSurface);
    geometryArrayPtr[1]->addPrimitiveSet(bottomSurface);
    geometryArrayPtr[2]->addPrimitiveSet(frontSurface);
    geometryArrayPtr[3]->addPrimitiveSet(backSurface);
    geometryArrayPtr[4]->addPrimitiveSet(leftSurface);
    geometryArrayPtr[5]->addPrimitiveSet(rightSurface);

    geometryArrayPtr[0]->addIndexCluster(0, 12, 20);
    geometryArrayPtr[0]->addIndexCluster(1, 13, 16);
    geometryArrayPtr[0]->addIndexCluster(2, 9, 17);
    geometryArrayPtr[0]->addIndexCluster(3, 8, 21);

    geometryArrayPtr[1]->addIndexCluster(4, 15, 23);
    geometryArrayPtr[1]->addIndexCluster(5, 14, 19);
    geometryArrayPtr[1]->addIndexCluster(6, 10, 18);
    geometryArrayPtr[1]->addIndexCluster(7, 11, 22);

    geometryArrayPtr[2]->addIndexCluster(3, 8, 21);
    geometryArrayPtr[2]->addIndexCluster(2, 9, 17);
    geometryArrayPtr[2]->addIndexCluster(6, 10, 18);
    geometryArrayPtr[2]->addIndexCluster(7, 11, 22);

    geometryArrayPtr[3]->addIndexCluster(0, 12, 20);
    geometryArrayPtr[3]->addIndexCluster(1, 13, 16);
    geometryArrayPtr[3]->addIndexCluster(5, 14, 19);
    geometryArrayPtr[3]->addIndexCluster(4, 15, 23);

    geometryArrayPtr[4]->addIndexCluster(1, 13, 16);
    geometryArrayPtr[4]->addIndexCluster(2, 9, 17);
    geometryArrayPtr[4]->addIndexCluster(6, 10, 18);
    geometryArrayPtr[4]->addIndexCluster(5, 14, 19);

    geometryArrayPtr[5]->addIndexCluster(0, 12, 20);
    geometryArrayPtr[5]->addIndexCluster(3, 8, 21);
    geometryArrayPtr[5]->addIndexCluster(7, 11, 22);
    geometryArrayPtr[5]->addIndexCluster(4, 15, 23);
}


/***************************************************************
* Function: initGeometryCylinder()
***************************************************************/
void CAVEGeodeShape::initGeometryCylinder(const Vec3 &initVect, const Vec3 &sVect)
{
    int numFanSegs = CAVEGeodeSnapWireframeCylinder::gCurFanSegments;
    float cx = initVect.x(), cy = initVect.y(), cz = initVect.z();
    float rad = sVect.x(), height = sVect.z();
    if (rad < 0) 
        rad = -rad;
    if (height < 0) 
    { 
        cz = initVect.z() + height;  
        height = -height; 
    }

    // take record of center vector and number of vertices, normals, texcoords
    mCenterVect = Vec3(cx, cy, cz + height * 0.5);
    mNumVertices = mNumNormals = mNumTexcoords = (numFanSegs + 1) * 4;

    // create vertical edges, cap radiating edges and ring strips on side surface
    float intvl = M_PI * 2 / numFanSegs;
    for (int i = 0; i <= numFanSegs; i++)
    {
        const float theta = i * intvl;
        const float cost = cos(theta);
        const float sint = sin(theta);

        mVertexArray->push_back(Vec3(cx, cy, cz) + Vec3(rad * cost, rad * sint, height));	// top surface
        mVertexArray->push_back(Vec3(cx, cy, cz) + Vec3(rad * cost, rad * sint, 0));		// bottom surface
        mVertexArray->push_back(Vec3(cx, cy, cz) + Vec3(rad * cost, rad * sint, height));	// upper side
        mVertexArray->push_back(Vec3(cx, cy, cz) + Vec3(rad * cost, rad * sint, 0));		// lower side

        mNormalArray->push_back(Vec3(0, 0, 1));		
        mNormalArray->push_back(Vec3(0, 0, -1));
        mNormalArray->push_back(Vec3(cost, sint, 0));	
        mNormalArray->push_back(Vec3(cost, sint, 0));	

        mUDirArray->push_back(Vec3(1, 0, 0));	mVDirArray->push_back(Vec3(0, 1, 0));		// top surface
        mUDirArray->push_back(Vec3(1, 0, 0));	mVDirArray->push_back(Vec3(0, 1, 0));		// bottom surface
        mUDirArray->push_back(Vec3(0, 0, 0));	mVDirArray->push_back(Vec3(0, 0, 1));		// upper side
        mUDirArray->push_back(Vec3(0, 0, 0));	mVDirArray->push_back(Vec3(0, 0, 1));		// lower side

        mTexcoordArray->push_back(Vec2(rad * cost, rad * sint) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(rad * cost, rad * sint) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(rad * intvl * i, height) / gTextureTileSize);
        mTexcoordArray->push_back(Vec2(rad * intvl * i, 0.0f) / gTextureTileSize);
    }


    // Snapping bounds
    float radius = 0.2;
    float snapSphereRadius = 0.2, snapSphereOpacity = 0.5;
    osg::Vec4 snapSphereColor = osg::Vec4(1, 0, 1, 0); 


    // Add edge bounding cylinders 
    std::vector<osg::Cylinder*> cylVec;
    float width = (mVertexArray->at(4) - mVertexArray->at(0)).length();

    for (int i = 0; i < numFanSegs; ++i)
    {
        // vertical edges
        osg::Vec3 center = mVertexArray->at(i*4) + osg::Vec3(0, 0, -height/2);
        osg::Cylinder *cyl = new osg::Cylinder(center, radius, height);
        cylVec.push_back(cyl);
        
        // horizontal edges
        float rot = intvl * i;
        center = mVertexArray->at(i*4 + 4) - mVertexArray->at(i*4);
        center[2] = 0;
        cyl = new osg::Cylinder(center, 0.1, width);
/*        cyl->setRotation(osg::Quat(0,      osg::Vec3(1, 0, 0),
                                   M_PI/2, osg::Vec3(0, 1, 0),
                                   rad*intvl*i,    osg::Vec3(0, 0, 1)));
                                   */
        cylVec.push_back(cyl);

        center[2] = height;
        cyl = new osg::Cylinder(center, 0.1, width);
/*        cyl->setRotation(osg::Quat(0,      osg::Vec3(1, 0, 0),
                                   M_PI/2, osg::Vec3(0, 1, 0),
                                   rad*intvl*i,    osg::Vec3(0, 0, 1)));
                                   */
        cylVec.push_back(cyl);

    }

    osg::StateSet *ss;
    osg::ShapeDrawable *shpDrawable;
    osg::Material *mat;
    osg::Geode *geode;
    for (int i = 0; i < cylVec.size(); ++i)
    {
        shpDrawable = new osg::ShapeDrawable(cylVec[i]);

        mat = new Material;
        mat->setDiffuse(Material::FRONT_AND_BACK, snapSphereColor);

        ss = shpDrawable->getOrCreateStateSet();
        ss->setMode(GL_BLEND, StateAttribute::PROTECTED | StateAttribute::ON );
        ss->setRenderingHint(StateAttribute::PROTECTED | StateSet::TRANSPARENT_BIN);
        ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
        ss->setMode(GL_CULL_FACE, StateAttribute::PROTECTED| StateAttribute::ON);

        addDrawable(shpDrawable);

        mEdgeDrawableMap[cylVec[i]] = shpDrawable; 
        geode = new osg::Geode();
        geode->addDrawable(shpDrawable);
        mEdgeGeodeMap[cylVec[i]] = geode;
    }


    // Add vertex bounding spheres
    for (int i = 0; i <= numFanSegs; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            osg::Sphere *sph = new osg::Sphere(mVertexArray->at((i*4) + j), snapSphereRadius);
            osg::ShapeDrawable *shpDraw = new osg::ShapeDrawable(sph);

            mat = new Material;
            mat->setDiffuse(Material::FRONT_AND_BACK, snapSphereColor);

            StateSet *ss = shpDraw->getOrCreateStateSet();
            ss->setMode(GL_BLEND, StateAttribute::PROTECTED | StateAttribute::ON );
            ss->setRenderingHint(StateAttribute::PROTECTED | StateSet::TRANSPARENT_BIN);
            ss->setAttributeAndModes(mat, StateAttribute::PROTECTED | StateAttribute::ON);
            ss->setMode(GL_CULL_FACE, StateAttribute::PROTECTED| StateAttribute::ON);

            addDrawable(shpDraw);
            mVertBoundingSpheres.push_back(sph);
            mShapeDrawableMap[sph] = shpDraw;
        }
    }


    // create geometries for each surface
    CAVEGeometry **geometryArrayPtr = new CAVEGeometry*[3];
    for (int i = 0; i < 3; i++)
    {
        geometryArrayPtr[i] = new CAVEGeometry;
        geometryArrayPtr[i]->setVertexArray(mVertexArray);
        geometryArrayPtr[i]->setNormalArray(mNormalArray);
        geometryArrayPtr[i]->setTexCoordArray(0, mTexcoordArray);
        geometryArrayPtr[i]->setNormalBinding(Geometry::BIND_PER_VERTEX);

        mGeometryVector.push_back(geometryArrayPtr[i]);
        addDrawable(geometryArrayPtr[i]);
    }

    // write primitive set and index clusters
    DrawElementsUInt* topSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);  
    DrawElementsUInt* bottomSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);

    for (int i = 0; i <= numFanSegs; i++)
    {
        topSurface->push_back(i * 4);
        bottomSurface->push_back((numFanSegs - i) * 4 + 1);
    }
    geometryArrayPtr[0]->addPrimitiveSet(topSurface);
    geometryArrayPtr[1]->addPrimitiveSet(bottomSurface);

    for (int i = 0; i < numFanSegs; i++)
    {
        DrawElementsUInt* sideSurface = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);
        sideSurface->push_back(i * 4 + 2);	sideSurface->push_back(i * 4 + 3);
        sideSurface->push_back(i * 4 + 7);	sideSurface->push_back(i * 4 + 6);
        geometryArrayPtr[2]->addPrimitiveSet(sideSurface);
    }

    for (int i = 0; i <= numFanSegs; i++)
    {
        geometryArrayPtr[0]->addIndexCluster(i * 4    , i * 4 + 2);
        geometryArrayPtr[1]->addIndexCluster(i * 4 + 1, i * 4 + 3);
        geometryArrayPtr[2]->addIndexCluster(i * 4    , i * 4 + 2);
        geometryArrayPtr[2]->addIndexCluster(i * 4 + 1, i * 4 + 3);
    }
}


/***************************************************************
* Function: applyEditorInfo()
***************************************************************/
void CAVEGeodeShape::applyEditorInfo(Vec3Array **vertexArrayPtr, Vec3Array **normalArrayPtr, 
		Vec3Array **udirArrayPtr, Vec3Array **vdirArrayPtr, Vec2Array **texcoordArrayPtr,
		const Vec3Array *refVertexArrayPtr, const Vec3Array *refNormalArrayPtr, 
		const Vec3Array *refUDirArrayPtr, const Vec3Array *refVDirArrayPtr, const Vec2Array *refTexcoordArrayPtr,
		const int &nVerts, EditorInfo **infoPtr, const VertexMaskingVector &vertMaskingVector)
{
    // access target and source data pointers
    Vec3 *geodeVertexDataPtr, *geodeNormalDataPtr, *geodeUDirDataPtr, *geodeVDirDataPtr;
    Vec2 *geodeTexcoordDataPtr;
    const Vec3 *refGeodeVertexDataPtr, *refGeodeNormalDataPtr, *refGeodeUDirDataPtr, *refGeodeVDirDataPtr;
    const Vec2 *refGeodeTexcoordDataPtr;

    geodeVertexDataPtr = (Vec3*) ((*vertexArrayPtr)->getDataPointer());
    geodeNormalDataPtr = (Vec3*) ((*normalArrayPtr)->getDataPointer());
    geodeUDirDataPtr = (Vec3*) ((*udirArrayPtr)->getDataPointer());
    geodeVDirDataPtr = (Vec3*) ((*vdirArrayPtr)->getDataPointer());
    geodeTexcoordDataPtr = (Vec2*) ((*texcoordArrayPtr)->getDataPointer());

    refGeodeVertexDataPtr = (const Vec3*) (refVertexArrayPtr->getDataPointer());
    refGeodeNormalDataPtr = (const Vec3*) (refNormalArrayPtr->getDataPointer());
    refGeodeUDirDataPtr =  (const Vec3*) (refUDirArrayPtr->getDataPointer());
    refGeodeVDirDataPtr =  (const Vec3*) (refVDirArrayPtr->getDataPointer());
    refGeodeTexcoordDataPtr = (const Vec2*) (refTexcoordArrayPtr->getDataPointer());

    /* implement vertex & normal updates with respect to all ActiveTypeMasking
       texture coordinates are only changed in 'MOVE' and 'SCALE' operations,
       texture directional vectors are only changed in 'ROTATE' operations. 
    */
    if ((*infoPtr)->getTypeMasking() == EditorInfo::MOVE)
    {
        const Vec3 offset = (*infoPtr)->getMoveOffset();
        for (int i = 0; i < nVerts; i++)
        {
            if (vertMaskingVector[i])
            {
                // apply offset values to vetex data vector
                geodeVertexDataPtr[i] = refGeodeVertexDataPtr[i] + offset;

                // apply offset values to texture coordinates, normal is not changed
                Vec3 udir = refGeodeUDirDataPtr[i];
                Vec3 vdir = refGeodeVDirDataPtr[i];
                Vec2 texoffset = Vec2(udir * offset, vdir * offset) / gTextureTileSize;
                geodeTexcoordDataPtr[i] = refGeodeTexcoordDataPtr[i] + texoffset;
            }
        }
    }

    else if ((*infoPtr)->getTypeMasking() == EditorInfo::ROTATE)
    {
        const Vec3 center = (*infoPtr)->getRotateCenter();
        const Vec3 axis = (*infoPtr)->getRotateAxis();
        const float angle = (*infoPtr)->getRotateAngle();

        Matrixf rotMat;
        rotMat.makeRotate(angle, axis);

        for (int i = 0; i < nVerts; i++)
        {
            if (vertMaskingVector[i])
            {
                // update vertex list: 'translation' -> 'rotation' -> 'reversed translation'
                Vec3 pos = refGeodeVertexDataPtr[i];
                geodeVertexDataPtr[i] = (pos - center) * rotMat + center;

                // update normal and u, v-direction vectors with single rotations
                Vec3 norm = refGeodeNormalDataPtr[i];
                Vec3 udir = refGeodeUDirDataPtr[i];
                Vec3 vdir = refGeodeVDirDataPtr[i];
                geodeNormalDataPtr[i] = norm * rotMat;
                geodeUDirDataPtr[i] = udir * rotMat;
                geodeVDirDataPtr[i] = vdir * rotMat;
            }
        }
    }

    else if ((*infoPtr)->getTypeMasking() == EditorInfo::SCALE)
    {
        const Vec3 center = (*infoPtr)->getScaleCenter();
        const Vec3 scale = (*infoPtr)->getScaleVect();

        Matrixf scaleMat;
        scaleMat.makeScale(scale);

        for (int i = 0; i < nVerts; i++)
        {
            if (vertMaskingVector[i])
            {
                // update vertex list: 'translation' -> 'scaling' -> 'reversed translation'
                Vec3 pos = refGeodeVertexDataPtr[i];
                geodeVertexDataPtr[i] = (pos - center) * scaleMat + center;
                Vec3 offset = geodeVertexDataPtr[i] - pos;

                // update texture coordinates 'u', 'v', normal and u, v-direction vectors are not changed
                Vec3 udir = refGeodeUDirDataPtr[i];
                Vec3 vdir = refGeodeVDirDataPtr[i];
                Vec2 texoffset = Vec2(udir * offset, vdir * offset) / gTextureTileSize;
                geodeTexcoordDataPtr[i] = refGeodeTexcoordDataPtr[i] + texoffset;
            }
        }
    }
    else return;
}

