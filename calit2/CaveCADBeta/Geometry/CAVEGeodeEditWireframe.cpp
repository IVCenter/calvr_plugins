/***************************************************************
* File Name: CAVEGeodeEditWireframe.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Jan 31, 2011
*
***************************************************************/
#include "CAVEGeodeEditWireframe.h"


using namespace std;
using namespace osg;


/* 'gSnappingUnitDist' is the default minimum sensible distance in CAVE geometry editting */
const float CAVEGeodeEditWireframe::gSnappingUnitDist(0.06f);


// Constructor: CAVEGeodeEditWireframe
CAVEGeodeEditWireframe::CAVEGeodeEditWireframe()
{
    mGeometry = new Geometry();
    addDrawable(mGeometry);

    Material* material = new Material;
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0, 1.0, 0.0, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.0, 1.0, 0.0, 1.0));
    material->setSpecular(Material::FRONT_AND_BACK, osg::Vec4( 1.f, 1.f, 1.f, 1.0f));
    material->setAlpha(Material::FRONT_AND_BACK, 1.f);

    StateSet* stateset = new StateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    stateset->setMode(GL_LIGHTING, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    setStateSet(stateset);
}


// Destructor: CAVEGeodeEditWireframe
CAVEGeodeEditWireframe::~CAVEGeodeEditWireframe()
{
}


/***************************************************************
* Function: initUnitWireBox
***************************************************************/
void CAVEGeodeEditWireframe::initUnitWireBox()
{
    float xMin, yMin, zMin, xMax, yMax, zMax;
    xMin = -0.5;	xMax = 0.5;
    yMin = -0.5;	yMax = 0.5;
    zMin = -0.5;	zMax = 0.5;

    Vec3Array* vertices = new Vec3Array;
    vertices->push_back(Vec3(xMax, yMax, zMax));
    vertices->push_back(Vec3(xMin, yMax, zMax));
    vertices->push_back(Vec3(xMin, yMin, zMax));
    vertices->push_back(Vec3(xMax, yMin, zMax)); 
    vertices->push_back(Vec3(xMax, yMax, zMin));
    vertices->push_back(Vec3(xMin, yMax, zMin));
    vertices->push_back(Vec3(xMin, yMin, zMin));
    vertices->push_back(Vec3(xMax, yMin, zMin));

    Vec3Array* normals = new Vec3Array;
    normals->push_back(Vec3(xMax, yMax, zMax));
    normals->push_back(Vec3(xMin, yMax, zMax));
    normals->push_back(Vec3(xMin, yMin, zMax));
    normals->push_back(Vec3(xMax, yMin, zMax));
    normals->push_back(Vec3(xMax, yMax, zMin));
    normals->push_back(Vec3(xMin, yMax, zMin));
    normals->push_back(Vec3(xMin, yMin, zMin));
    normals->push_back(Vec3(xMax, yMin, zMin));


    DrawElementsUInt* topEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);  
    DrawElementsUInt* bottomEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    topEdges->push_back(0);    	bottomEdges->push_back(4);
    topEdges->push_back(1);    	bottomEdges->push_back(5);
    topEdges->push_back(2);    	bottomEdges->push_back(6);
    topEdges->push_back(3);    	bottomEdges->push_back(7);
    topEdges->push_back(0);    	bottomEdges->push_back(4);

    DrawElementsUInt* sideEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);  
    sideEdges->push_back(0);   	sideEdges->push_back(4);
    sideEdges->push_back(1);   	sideEdges->push_back(5);
    sideEdges->push_back(2);   	sideEdges->push_back(6);
    sideEdges->push_back(3);   	sideEdges->push_back(7);

    mGeometry = new Geometry;
    mGeometry->setVertexArray(vertices);
    mGeometry->setNormalArray(normals);
    mGeometry->addPrimitiveSet(topEdges);
    mGeometry->addPrimitiveSet(bottomEdges);
    mGeometry->addPrimitiveSet(sideEdges);
    addDrawable(mGeometry);
}


/***************************************************************
* Constructor: CAVEGeodeEditWireframeMove
*
* Default wireframe is unit-sized box with regular 12 edges
* Actual sizes of the wireframe is decided by matrix transforms
* and PA transform in higher level of scene graph tree.
*
***************************************************************/
CAVEGeodeEditWireframeMove::CAVEGeodeEditWireframeMove()
{
    initUnitWireBox();
}


/***************************************************************
* Constructor: CAVEGeodeEditWireframeRotate
*
* Default wireframe is composed of three unit sized circles in
* XY, YZ, XZ planes respectively. Each circle has 36 segments
*
***************************************************************/
CAVEGeodeEditWireframeRotate::CAVEGeodeEditWireframeRotate()
{
    const int numSegs = 36;
    const float stepTheta = M_PI * 2 / numSegs;

    /* create 72 vertices and normals in three planes */
    Vec3Array* vertices = new Vec3Array;
    Vec3Array* normals = new Vec3Array;

    for (int i = 0; i < numSegs; i++)
    {
	float theta = stepTheta * i;
	float cosT = cos(theta);
	float sinT = sin(theta);

	vertices->push_back(Vec3(cosT, sinT, 0));	// vertex in XY plane
	vertices->push_back(Vec3(cosT, 0, sinT));	// vertex in XZ plane
	vertices->push_back(Vec3(0, cosT, sinT));	// vertex in YZ plane

	normals->push_back(Vec3(cosT, sinT, 0));	// vertex in XY plane
	normals->push_back(Vec3(cosT, 0, sinT));	// vertex in XZ plane
	normals->push_back(Vec3(0, cosT, sinT));	// vertex in YZ plane
    }

    DrawElementsUInt* xyPlaneEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    DrawElementsUInt* yzPlaneEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    DrawElementsUInt* xzPlaneEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    for (int i = 0; i < numSegs; i++)
    {
	xyPlaneEdges->push_back(i * 3);
	yzPlaneEdges->push_back(i * 3 + 1);
	xzPlaneEdges->push_back(i * 3 + 2);
    }
    xyPlaneEdges->push_back(0);
    yzPlaneEdges->push_back(1);
    xzPlaneEdges->push_back(2);

    mGeometry = new Geometry;
    mGeometry->setVertexArray(vertices);
    mGeometry->setNormalArray(normals);
    mGeometry->addPrimitiveSet(xyPlaneEdges);
    mGeometry->addPrimitiveSet(yzPlaneEdges);
    mGeometry->addPrimitiveSet(xzPlaneEdges);
    addDrawable(mGeometry);
}


/***************************************************************
* Constructor: CAVEGeodeEditWireframeManipulate
***************************************************************/
CAVEGeodeEditWireframeManipulate::CAVEGeodeEditWireframeManipulate()
{
    initUnitWireBox();
}


/***************************************************************
* Constructor: CAVEGeodeEditGeometryWireframe
*
* Geode object 'CAVEGeodeEditGeometryWireframe' does have the
* field of vertex, normal or texture coordinate arrays. It just
* takes references for all those from the 'CAVEGeometry', then
* creates a wireframe geometry according to the primitive sets
* defined in 'CAVEGeometry' without any solid surface.
*
***************************************************************/
CAVEGeodeEditGeometryWireframe::CAVEGeodeEditGeometryWireframe(CAVEGeometry *geometry)
{
    Array *geodeVertexArray = geometry->getVertexArray();
    Array *geodeNormalArray = geometry->getNormalArray(); 
    Array *geodeTexcoordArray = geometry->getTexCoordArray(0);

    /* clone the field of primitive sets, set all modes to 'LINE_STRIP' */
    mCAVEGeometry = new CAVEGeometry(geometry);
    mCAVEGeometry->setPrimitiveSetModes(PrimitiveSet::LINE_STRIP);
    addDrawable(mCAVEGeometry);

    /* check the valid status of all data field from 'CAVEGeodeShape', refer them to 'this' geode */
    if (geodeVertexArray->getType() == Array::Vec3ArrayType)
	mCAVEGeometry->setVertexArray((Vec3Array*) geodeVertexArray);
    else return;

    if (geodeNormalArray->getType() == Array::Vec3ArrayType)
	mCAVEGeometry->setNormalArray((Vec3Array*) geodeNormalArray);
    else return;

    if (geodeTexcoordArray->getType() == Array::Vec2ArrayType)
        mCAVEGeometry->setTexCoordArray(0, (Vec2Array*) geodeTexcoordArray);
    else return;

    mCAVEGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);
}













