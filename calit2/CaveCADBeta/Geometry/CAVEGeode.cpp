/***************************************************************
* File Name: CAVEGeode.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Sep 13, 2010
*
***************************************************************/
#include "CAVEGeode.h"


using namespace std;
using namespace osg;


//Constructor
CAVEGeode::CAVEGeode(): mAlpha(1.0f), mDiffuse(Vec3(1, 1, 1)), mSpecular(Vec3(0, 0, 0))
{
    mTexFilename = getDataDir() + "Textures/White.JPG";
}

//Destructor
CAVEGeode::~CAVEGeode()
{
}

const string CAVEGeode::getDataDir() { return string("/home/covise/data/CaveCADBeta/"); }


/***************************************************************
* Function: applyColorTexture()
***************************************************************/
void CAVEGeode::applyColorTexture(const Vec3 &diffuse, const Vec3 &specular, const float &alpha, const string &texFilename)
{
    StateSet *stateset = getOrCreateStateSet();

    /* update material parameters */
    Material *material = dynamic_cast <Material*> (stateset->getAttribute(StateAttribute::MATERIAL)); 
    if (!material) material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuse, 1.f));
    material->setSpecular(Material::FRONT_AND_BACK, Vec4(specular, 1.f));
    material->setAlpha(Material::FRONT_AND_BACK, alpha);
    stateset->setAttributeAndModes(material, StateAttribute::ON);

    /* update texture image */
    Texture2D *texture = dynamic_cast <Texture2D*> (stateset->getAttribute(StateAttribute::TEXTURE));
    if (!texture) texture = new Texture2D;
    Image* texImage = osgDB::readImageFile(texFilename);
    texture->setImage(texImage); 
    texture->setWrap(Texture::WRAP_S,Texture::MIRROR);
    texture->setWrap(Texture::WRAP_T,Texture::MIRROR);
    texture->setDataVariance(Object::DYNAMIC);
    stateset->setTextureAttributeAndModes(0, texture, StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

    /* update of color & texture properties */
    mDiffuse = diffuse;
    mSpecular = specular;
    mAlpha = alpha;
    mTexFilename = texFilename;
}


/***************************************************************
* Function: applyAlpha()
***************************************************************/
void CAVEGeode::applyAlpha(const float &alpha)
{
    StateSet *stateset = getOrCreateStateSet();

    /* update material parameters */
    Material *material = dynamic_cast <Material*> (stateset->getAttribute(StateAttribute::MATERIAL)); 
    if (!material) material = new Material;
    material->setAlpha(Material::FRONT_AND_BACK, alpha);
    stateset->setAttributeAndModes(material, StateAttribute::ON);

    /* update alpha value */
    mAlpha = alpha;
}


















