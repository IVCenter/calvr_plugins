/***************************************************************
* File Name: CAVEGeodeSnapWireframe.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Nov 17, 2010
*
***************************************************************/
#include "CAVEGeodeSnapWireframe.h"


using namespace std;
using namespace osg;


// 'gSnappingUnitDist' is the default minimum sensible distance in CAVE design space, or the size of unit grid. 
//  could define different snapping unit distance in other derived classes of CAVEGeode
const float CAVEGeodeSnapWireframe::gSnappingUnitDist(0.010f);

const int CAVEGeodeSnapWireframeCylinder::gMinFanSegments(18);
int CAVEGeodeSnapWireframeCylinder::gCurFanSegments(18);

const int CAVEGeodeSnapWireframeCone::gMinFanSegments(18);
int CAVEGeodeSnapWireframeCone::gCurFanSegments(18);


// Constructor: CAVEGeodeSnapWireframe
CAVEGeodeSnapWireframe::CAVEGeodeSnapWireframe()
{
    // unit grid size 'mSnappingUnitDist' will inherit the default value 'gSnappingUnitDist' unless modified
    mSnappingUnitDist = gSnappingUnitDist;

    mInitPosition = Vec3(0, 0, 0);
    mScaleVect = Vec3(1, 1, 1);
    mDiagonalVect = Vec3(1, 1, 1);

    mBaseGeometry = new Geometry();
    mSnapwireGeometry = new Geometry();
    addDrawable(mBaseGeometry);
    addDrawable(mSnapwireGeometry);

    Material* material = new Material;
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0, 1.0, 0.0, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 0.0, 1.0));
    material->setSpecular(Material::FRONT_AND_BACK, osg::Vec4( 1.f, 1.f, 1.f, 1.0f));
    material->setAlpha(Material::FRONT_AND_BACK, 0.f);

    StateSet* stateset = new StateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    stateset->setMode(GL_LIGHTING, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    setStateSet(stateset);
}

// Destructor: CAVEGeodeSnapWireframe
CAVEGeodeSnapWireframe::~CAVEGeodeSnapWireframe()
{
}


// Constructor: CAVEGeodeSnapWireframeBox
CAVEGeodeSnapWireframeBox::CAVEGeodeSnapWireframeBox()
{
    initBaseGeometry();
}

// Constructor: CAVEGeodeSnapWireframeCylinder
CAVEGeodeSnapWireframeCylinder::CAVEGeodeSnapWireframeCylinder()
{
    initBaseGeometry();
}

// Constructor: CAVEGeodeSnapWireframeCone
CAVEGeodeSnapWireframeCone::CAVEGeodeSnapWireframeCone()
{
    initBaseGeometry();
}


// Box

/***************************************************************
* Function: initBaseGeometry()
***************************************************************/
void CAVEGeodeSnapWireframeBox::initBaseGeometry()
{
    float xMin, yMin, zMin, xMax, yMax, zMax;
    xMin = 0.0;		xMax = 1.0;
    yMin = 0.0;		yMax = 1.0;
    zMin = 0.0;		zMax = 1.0;

    Vec3Array* vertices = new Vec3Array;
    vertices->push_back(Vec3(xMax, yMax, zMax));
    vertices->push_back(Vec3(xMin, yMax, zMax));
    vertices->push_back(Vec3(xMin, yMin, zMax));
    vertices->push_back(Vec3(xMax, yMin, zMax)); 
    vertices->push_back(Vec3(xMax, yMax, zMin));
    vertices->push_back(Vec3(xMin, yMax, zMin));
    vertices->push_back(Vec3(xMin, yMin, zMin));
    vertices->push_back(Vec3(xMax, yMin, zMin));
    mBaseGeometry->setVertexArray(vertices);

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

    mBaseGeometry->addPrimitiveSet(topEdges);
    mBaseGeometry->addPrimitiveSet(bottomEdges);
    mBaseGeometry->addPrimitiveSet(sideEdges);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeSnapWireframeBox::resize(osg::Vec3 &gridVect)
{
    /*
    Material* material = new Material;
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.0, 1.0, 0.0, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 0.0, 1.0));
    material->setSpecular(Material::FRONT_AND_BACK, osg::Vec4( 1.f, 1.f, 1.f, 1.0f));
    material->setAlpha(Material::FRONT_AND_BACK, 1.f);

    StateSet* stateset = new StateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    setStateSet(stateset);
    */
    
    // calculate grid vector 
    float snapUnitX, snapUnitY, snapUnitZ;
    snapUnitX = snapUnitY = snapUnitZ = mSnappingUnitDist;
    if (mScaleVect.x() < 0) 
        snapUnitX = -mSnappingUnitDist;
    if (mScaleVect.y() < 0) 
        snapUnitY = -mSnappingUnitDist;
    if (mScaleVect.z() < 0) 
        snapUnitZ = -mSnappingUnitDist;
    int xSeg = (int)(abs((int)((mScaleVect.x() + 0.5 * snapUnitX) / mSnappingUnitDist)));
    int ySeg = (int)(abs((int)((mScaleVect.y() + 0.5 * snapUnitY) / mSnappingUnitDist)));
    int zSeg = (int)(abs((int)((mScaleVect.z() + 0.5 * snapUnitZ) / mSnappingUnitDist)));

    Vec3 roundedVect;
    roundedVect.x() = xSeg * snapUnitX;		gridVect.x() = roundedVect.x() / mSnappingUnitDist;
    roundedVect.y() = ySeg * snapUnitY;		gridVect.y() = roundedVect.y() / mSnappingUnitDist;
    roundedVect.z() = zSeg * snapUnitZ;		gridVect.z() = roundedVect.z() / mSnappingUnitDist;
    mDiagonalVect = roundedVect;

    // update box corners in 'mBaseGeometry'
    float xMin, yMin, zMin, xMax, yMax, zMax;
    xMin = mInitPosition.x();	xMax = xMin + roundedVect.x();
    yMin = mInitPosition.y();	yMax = yMin + roundedVect.y();
    zMin = mInitPosition.z();	zMax = zMin + roundedVect.z();

    Array* baseVertArray = mBaseGeometry->getVertexArray();
    if (baseVertArray->getType() == Array::Vec3ArrayType)
    {
        Vec3* vertexArrayDataPtr = (Vec3*) (baseVertArray->getDataPointer());
        vertexArrayDataPtr[0] = Vec3(xMax, yMax, zMax);
        vertexArrayDataPtr[1] = Vec3(xMin, yMax, zMax);
        vertexArrayDataPtr[2] = Vec3(xMin, yMin, zMax);
        vertexArrayDataPtr[3] = Vec3(xMax, yMin, zMax);
        vertexArrayDataPtr[4] = Vec3(xMax, yMax, zMin);
        vertexArrayDataPtr[5] = Vec3(xMin, yMax, zMin);
        vertexArrayDataPtr[6] = Vec3(xMin, yMin, zMin);
        vertexArrayDataPtr[7] = Vec3(xMax, yMin, zMin);
    }
    mBaseGeometry->dirtyDisplayList();
    mBaseGeometry->dirtyBound();

    // update snapping wire geometry
    if (mSnapwireGeometry) 
        removeDrawable(mSnapwireGeometry);

    mSnapwireGeometry = new Geometry();
    Vec3Array* snapvertices = new Vec3Array;
    int vertoffset = 0;
    if (xSeg > 1)	// (xSeg - 1) * 4 vertices
    {
        for (int i = 1; i <= xSeg - 1; i++)
        {
            snapvertices->push_back(Vec3(xMin + i * snapUnitX, yMin, zMin));
            snapvertices->push_back(Vec3(xMin + i * snapUnitX, yMax, zMin));
            snapvertices->push_back(Vec3(xMin + i * snapUnitX, yMax, zMax));
            snapvertices->push_back(Vec3(xMin + i * snapUnitX, yMin, zMax));

            DrawElementsUInt* edges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            edges->push_back(vertoffset + (i-1)*4);
            edges->push_back(vertoffset + (i-1)*4 + 1);
            edges->push_back(vertoffset + (i-1)*4 + 2);
            edges->push_back(vertoffset + (i-1)*4 + 3);
            edges->push_back(vertoffset + (i-1)*4);
            mSnapwireGeometry->addPrimitiveSet(edges);
        }
        vertoffset += (xSeg - 1) * 4;
    }
    if (ySeg > 1)	// (ySeg - 1) * 4 vertices
    {
        for (int i = 1; i <= ySeg - 1; i++)
        {
            snapvertices->push_back(Vec3(xMin, yMin + i * snapUnitY, zMin));
            snapvertices->push_back(Vec3(xMax, yMin + i * snapUnitY, zMin));
            snapvertices->push_back(Vec3(xMax, yMin + i * snapUnitY, zMax));
            snapvertices->push_back(Vec3(xMin, yMin + i * snapUnitY, zMax));

            DrawElementsUInt* edges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            edges->push_back(vertoffset + (i-1)*4);
            edges->push_back(vertoffset + (i-1)*4 + 1);
            edges->push_back(vertoffset + (i-1)*4 + 2);
            edges->push_back(vertoffset + (i-1)*4 + 3);
            edges->push_back(vertoffset + (i-1)*4);
            mSnapwireGeometry->addPrimitiveSet(edges);
        }
        vertoffset += (ySeg - 1) * 4;
    }
    if (zSeg > 1)	// (zSeg - 1) * 4 vertices
    {
        for (int i = 1; i <= zSeg - 1; i++)
        {
            snapvertices->push_back(Vec3(xMin, yMin, zMin + i * snapUnitZ));
            snapvertices->push_back(Vec3(xMax, yMin, zMin + i * snapUnitZ));
            snapvertices->push_back(Vec3(xMax, yMax, zMin + i * snapUnitZ));
            snapvertices->push_back(Vec3(xMin, yMax, zMin + i * snapUnitZ));

            DrawElementsUInt* edges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
            edges->push_back(vertoffset + (i-1)*4);
            edges->push_back(vertoffset + (i-1)*4 + 1);
            edges->push_back(vertoffset + (i-1)*4 + 2);
            edges->push_back(vertoffset + (i-1)*4 + 3);
            edges->push_back(vertoffset + (i-1)*4);
            mSnapwireGeometry->addPrimitiveSet(edges);
        }
    }
    mSnapwireGeometry->setVertexArray(snapvertices);
    addDrawable(mSnapwireGeometry);
}


// Cylinder

/***************************************************************
* Function: initBaseGeometry()
***************************************************************/
void CAVEGeodeSnapWireframeCylinder::initBaseGeometry()
{
    Vec3Array* vertices = new Vec3Array;
    float rad = 1.0f, height = 1.0f, intvl = M_PI * 2 / gMinFanSegments;

    // BaseGeometry contains (gMinFanSegments * 2) vertices
    for (int i = 0; i < gMinFanSegments; i++) 
    {
        vertices->push_back(Vec3(rad * cos(i * intvl), rad * sin(i * intvl), height));
    }
    for (int i = 0; i < gMinFanSegments; i++) 
    {
        vertices->push_back(Vec3(rad * cos(i * intvl), rad * sin(i * intvl), 0));
    }
    mBaseGeometry->setVertexArray(vertices);

    DrawElementsUInt* topEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    DrawElementsUInt* bottomEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    DrawElementsUInt* sideEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    for (int i = 0; i < gMinFanSegments; i++)
    {
        topEdges->push_back(i);
        bottomEdges->push_back(i + gMinFanSegments);
        sideEdges->push_back(i);
        sideEdges->push_back(i + gMinFanSegments);
    }
    topEdges->push_back(0);
    bottomEdges->push_back(gMinFanSegments);

    mBaseGeometry->addPrimitiveSet(topEdges);
    mBaseGeometry->addPrimitiveSet(bottomEdges);
    mBaseGeometry->addPrimitiveSet(sideEdges);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeSnapWireframeCylinder::resize(osg::Vec3 &gridVect)
{
    // calculate rounded vector
    float height = mScaleVect.z(), 
          rad = sqrt(mScaleVect.x() * mScaleVect.x() + mScaleVect.y() * mScaleVect.y());
    int hSeg, radSeg, fanSeg, hDir = 1;
    if (height < 0) 
    { 
        height = -height;  
        hDir = -1; 
    }

    hSeg = (int)(abs((int)(height / mSnappingUnitDist)) + 0.5);
    radSeg = (int)(abs((int)(rad / mSnappingUnitDist)) + 0.5);
    fanSeg = (int)(abs((int)(rad * M_PI * 2 / mSnappingUnitDist)) + 0.5);

    if (fanSeg < gMinFanSegments) 
        fanSeg = gMinFanSegments;

    float intvl = M_PI * 2 / fanSeg;
    height = hSeg * mSnappingUnitDist;
    rad = radSeg * mSnappingUnitDist;
    gridVect = Vec3(radSeg, 0, hSeg * hDir);

    mDiagonalVect = Vec3(rad, rad, height * hDir);

    gCurFanSegments = 10;//fanSeg;	// update number of fan segment, this parameter is passed to 'CAVEGeodeShape'

    // update 'mSnapwireGeometry' geometry, do not use 'mBaseGeometry' anymore
    if (mBaseGeometry)
    {
        removeDrawable(mBaseGeometry);
        mBaseGeometry = NULL;
    }
    if (mSnapwireGeometry) 
        removeDrawable(mSnapwireGeometry);

    mSnapwireGeometry = new Geometry();
    Vec3Array* snapvertices = new Vec3Array;
    int vertoffset = 0;

    // create vertical edges, cap radiating edges and ring strips on side surface
    for (int i = 0; i <= hSeg; i++)
    {
        for (int j = 0; j < fanSeg; j++)
        {
            float theta = j * intvl;
            snapvertices->push_back(mInitPosition + Vec3(rad * cos(theta), rad * sin(theta), i * mSnappingUnitDist * hDir));
        }
    }
    snapvertices->push_back(mInitPosition);
    snapvertices->push_back(mInitPosition + Vec3(0, 0, height * hDir));

    for (int i = 0; i <= hSeg; i++)
    {
        DrawElementsUInt* sideRingStrip = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
        for (int j = 0; j < fanSeg; j++) 
        {
            sideRingStrip->push_back(vertoffset + i * fanSeg + j);
        }
        sideRingStrip->push_back(vertoffset + i * fanSeg);
        mSnapwireGeometry->addPrimitiveSet(sideRingStrip);
    }
    DrawElementsUInt* sideVerticalEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    DrawElementsUInt* capRadiatingEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);

    for (int j = 0; j < fanSeg; j++)
    {
        sideVerticalEdges->push_back(vertoffset + j);
        sideVerticalEdges->push_back(vertoffset + j + fanSeg * hSeg);

        capRadiatingEdges->push_back((hSeg + 1) * fanSeg);
        capRadiatingEdges->push_back(vertoffset + j);
        capRadiatingEdges->push_back((hSeg + 1) * fanSeg + 1);
        capRadiatingEdges->push_back(vertoffset + j + fanSeg * hSeg);
    }
    mSnapwireGeometry->addPrimitiveSet(sideVerticalEdges);
    mSnapwireGeometry->addPrimitiveSet(capRadiatingEdges);
    vertoffset += (hSeg + 1) * fanSeg + 2;

    // create ring strips on two caps
    for (int i = 1; i < radSeg; i++)
    {
        float r = i * mSnappingUnitDist;
        for (int j = 0; j < fanSeg; j++)
        {
            float theta = j * intvl;
            snapvertices->push_back(mInitPosition + Vec3(r * cos(theta), r * sin(theta), height * hDir));
            snapvertices->push_back(mInitPosition + Vec3(r * cos(theta), r * sin(theta), 0));
        }
    }

    for (int i = 1; i < radSeg; i++)
    {
        DrawElementsUInt* topRingStrip = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
        DrawElementsUInt* bottomRingStrip = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);

        for (int j = 0; j < fanSeg; j++)
        {
            topRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2 + j * 2);
            bottomRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2 + j * 2 + 1);
        }
        topRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2);
        bottomRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2 + 1);
        mSnapwireGeometry->addPrimitiveSet(topRingStrip);
        mSnapwireGeometry->addPrimitiveSet(bottomRingStrip);
    }
    vertoffset += (radSeg - 1) * fanSeg * 2;

    mSnapwireGeometry->setVertexArray(snapvertices);
    addDrawable(mSnapwireGeometry);
}


// Cone

/***************************************************************
* Function: initBaseGeometry()
***************************************************************/
void CAVEGeodeSnapWireframeCone::initBaseGeometry()
{
    Vec3Array* vertices = new Vec3Array;
    float rad = 1.0f, height = 1.0f, intvl = M_PI * 2 / gMinFanSegments;

    // BaseGeometry contains (gMinFanSegments * 2) vertices
    for (int i = 0; i < gMinFanSegments; i++) 
    {
        vertices->push_back(Vec3(rad * cos(i * intvl), rad * sin(i * intvl), height));
    }
    for (int i = 0; i < gMinFanSegments; i++) 
    {
        vertices->push_back(Vec3(rad * cos(i * intvl), rad * sin(i * intvl), 0));
    }
    mBaseGeometry->setVertexArray(vertices);

    DrawElementsUInt* topEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    DrawElementsUInt* bottomEdges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
    DrawElementsUInt* sideEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    for (int i = 0; i < gMinFanSegments; i++)
    {
        topEdges->push_back(i);
        bottomEdges->push_back(i + gMinFanSegments);
        sideEdges->push_back(i);
        sideEdges->push_back(i + gMinFanSegments);
    }
    topEdges->push_back(0);
    bottomEdges->push_back(gMinFanSegments);

    mBaseGeometry->addPrimitiveSet(topEdges);
    mBaseGeometry->addPrimitiveSet(bottomEdges);
    mBaseGeometry->addPrimitiveSet(sideEdges);
}


/***************************************************************
* Function: resize()
***************************************************************/
void CAVEGeodeSnapWireframeCone::resize(osg::Vec3 &gridVect)
{
    // calculate rounded vector
    float height = mScaleVect.z(), 
          rad = sqrt(mScaleVect.x() * mScaleVect.x() + mScaleVect.y() * mScaleVect.y());
    int hSeg, radSeg, fanSeg, hDir = 1;
    if (height < 0) 
    { 
        height = -height;  
        hDir = -1; 
    }

    hSeg = (int)(abs((int)(height / mSnappingUnitDist)) + 0.5);
    radSeg = (int)(abs((int)(rad / mSnappingUnitDist)) + 0.5);
    fanSeg = (int)(abs((int)(rad * M_PI * 2 / mSnappingUnitDist)) + 0.5);

    if (fanSeg < gMinFanSegments) 
        fanSeg = gMinFanSegments;

    float intvl = M_PI * 2 / fanSeg;
    height = hSeg * mSnappingUnitDist;
    rad = radSeg * mSnappingUnitDist;
    gridVect = Vec3(radSeg, 0, hSeg * hDir);

    mDiagonalVect = Vec3(rad, rad, height * hDir);

    gCurFanSegments = 10;//fanSeg;	// update number of fan segment, this parameter is passed to 'CAVEGeodeShape'

    // update 'mSnapwireGeometry' geometry, do not use 'mBaseGeometry' anymore
    if (mBaseGeometry)
    {
        removeDrawable(mBaseGeometry);
        mBaseGeometry = NULL;
    }
    if (mSnapwireGeometry) 
        removeDrawable(mSnapwireGeometry);

    mSnapwireGeometry = new Geometry();
    Vec3Array* snapvertices = new Vec3Array;
    int vertoffset = 0;

    // create vertical edges, cap radiating edges and ring strips on side surface
    for (int i = 0; i <= hSeg; i++)
    {
        for (int j = 0; j < fanSeg; j++)
        {
            float theta = j * intvl;
            snapvertices->push_back(mInitPosition + Vec3(rad * cos(theta), rad * sin(theta), i * mSnappingUnitDist * hDir));
        }
    }
    snapvertices->push_back(mInitPosition);
    snapvertices->push_back(mInitPosition + Vec3(0, 0, height * hDir));

    for (int i = 0; i <= hSeg; i++)
    {
        DrawElementsUInt* sideRingStrip = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
        for (int j = 0; j < fanSeg; j++) 
        {
            sideRingStrip->push_back(vertoffset + i * fanSeg + j);
        }
        sideRingStrip->push_back(vertoffset + i * fanSeg);
        mSnapwireGeometry->addPrimitiveSet(sideRingStrip);
    }
    DrawElementsUInt* sideVerticalEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    DrawElementsUInt* capRadiatingEdges = new DrawElementsUInt(PrimitiveSet::LINES, 0);

    for (int j = 0; j < fanSeg; j++)
    {
        sideVerticalEdges->push_back(vertoffset + j);
        sideVerticalEdges->push_back(vertoffset + j + fanSeg * hSeg);

        capRadiatingEdges->push_back((hSeg + 1) * fanSeg);
        capRadiatingEdges->push_back(vertoffset + j);
        capRadiatingEdges->push_back((hSeg + 1) * fanSeg + 1);
        capRadiatingEdges->push_back(vertoffset + j + fanSeg * hSeg);
    }
    mSnapwireGeometry->addPrimitiveSet(sideVerticalEdges);
    mSnapwireGeometry->addPrimitiveSet(capRadiatingEdges);
    vertoffset += (hSeg + 1) * fanSeg + 2;

    // create ring strips on two caps
    for (int i = 1; i < radSeg; i++)
    {
        float r = i * mSnappingUnitDist;
        for (int j = 0; j < fanSeg; j++)
        {
            float theta = j * intvl;
            snapvertices->push_back(mInitPosition + Vec3(r * cos(theta), r * sin(theta), height * hDir));
            snapvertices->push_back(mInitPosition + Vec3(r * cos(theta), r * sin(theta), 0));
        }
    }

    for (int i = 1; i < radSeg; i++)
    {
        DrawElementsUInt* topRingStrip = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
        DrawElementsUInt* bottomRingStrip = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);

        for (int j = 0; j < fanSeg; j++)
        {
            topRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2 + j * 2);
            bottomRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2 + j * 2 + 1);
        }
        topRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2);
        bottomRingStrip->push_back(vertoffset + (i-1) * fanSeg * 2 + 1);
        mSnapwireGeometry->addPrimitiveSet(topRingStrip);
        mSnapwireGeometry->addPrimitiveSet(bottomRingStrip);
    }
    vertoffset += (radSeg - 1) * fanSeg * 2;

    mSnapwireGeometry->setVertexArray(snapvertices);
    addDrawable(mSnapwireGeometry);
}

