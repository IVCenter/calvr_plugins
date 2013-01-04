/***************************************************************
* File Name: CAVEGeodeIconSurface.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Jan 19, 2011
*
***************************************************************/
#include "CAVEGeodeIconSurface.h"


using namespace std;
using namespace osg;


/* default colors and transparencies */
const float CAVEGeodeIconSurface::gAlphaNormal(0.4f);
const float CAVEGeodeIconSurface::gAlphaSelected(0.9f);
const float CAVEGeodeIconSurface::gAlphaUnselected(0.1f);
const Vec3 CAVEGeodeIconSurface::gDiffuseColorNormal(Vec3(1, 1, 1));
const Vec3 CAVEGeodeIconSurface::gDiffuseColorSelected(Vec3(1, 0.5, 0));
const Vec3 CAVEGeodeIconSurface::gDiffuseColorUnselected(Vec3(1, 1, 1));
const Vec3 CAVEGeodeIconSurface::gSpecularColor(Vec3(0, 0, 0));

//Constructor
CAVEGeodeIconSurface::CAVEGeodeIconSurface(Vec3Array **surfVertexArray, Vec3Array **surfNormalArray,
			Vec2Array **surfTexcoordArray, CAVEGeometry **surfGeometry, CAVEGeometry **surfGeometryRef)
{
    mGeometryOriginPtr = *surfGeometry;
    mGeometryReferencePtr = *surfGeometryRef;

    /* make a deep copy of the CAVEGeometry from CAVEGeodeShape */
    mGeometry = new CAVEGeometry(*surfGeometry);
    addDrawable(mGeometry);

    /* share vertex, normal and texture coordinate arrays of the same geode */
    mGeometry->setVertexArray(*surfVertexArray);
    mGeometry->setNormalArray(*surfNormalArray);
    mGeometry->setTexCoordArray(0, *surfTexcoordArray);
    mGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);

    /* apply color texture to virtual surface */
    applyColor(gDiffuseColorNormal, gSpecularColor, gAlphaNormal);
    applyTexture("");
}

//Destructor
CAVEGeodeIconSurface::~CAVEGeodeIconSurface()
{
}


/***************************************************************
* Function: setHighlightNormal
***************************************************************/
void CAVEGeodeIconSurface::setHighlightNormal()
{
    applyColor(gDiffuseColorNormal, gSpecularColor, gAlphaNormal);
    applyTexture("");
}


/***************************************************************
* Function: setHighlightSelected
***************************************************************/
void CAVEGeodeIconSurface::setHighlightSelected()
{
    applyColor(gDiffuseColorSelected, gSpecularColor, gAlphaSelected);
    applyTexture("");
}


/***************************************************************
* Function: setHighlightUnselected
***************************************************************/
void CAVEGeodeIconSurface::setHighlightUnselected()
{
    applyColor(gDiffuseColorUnselected, gSpecularColor, gAlphaUnselected);
    applyTexture("");
}

