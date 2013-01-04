/***************************************************************
* Animation File Name: ANIMRefSkyDome.cpp
*
* Description: Create reference sky dome
*
* Written by ZHANG Lelin on Sep 29, 2010
*
***************************************************************/
#include "ANIMRefSkyDome.h"

using namespace std;
using namespace osg;


namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMCreateRefSkyDome()
*
***************************************************************/
MatrixTransform *ANIMCreateRefSkyDome(StateSet **stateset)
{
    /* sky dome geometry */
    Sphere *skyShape = new Sphere();
    ShapeDrawable* skyDrawable = new ShapeDrawable(skyShape);
    Geode* skyGeode = new Geode();
    MatrixTransform *skyDomeTrans = new MatrixTransform;
    
    //osg::Matrix m;
    //m.makeRotate(osg::Quat(M_PI/2, osg::Vec3(0, 1, 0)));
    //skyDomeTrans->setMatrix(m);

    skyShape->setRadius(ANIM_SKYDOME_RADIUS);
    skyGeode->addDrawable(skyDrawable);
    skyDomeTrans->addChild(skyGeode);

    // apply simple colored materials
    Material* material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0f);

    (*stateset) = new StateSet();
    (*stateset)->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    (*stateset)->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    (*stateset)->setRenderingHint(StateSet::TRANSPARENT_BIN);
    skyGeode->setStateSet(*stateset);

//    skyGeode->setNodeMask(0xFFFFFF & ~(0x2 | 0x3));

    // load sky dome shader
    Uniform* sunrUniform = new Uniform("hazeRadisuMin", 0.975f);
    (*stateset)->addUniform(sunrUniform);

    Uniform* sunRUniform = new Uniform("hazeRadisuMax", 0.995f);
    (*stateset)->addUniform(sunRUniform);

    Uniform* sunDirUniform = new Uniform("sundir", Vec4(0.0, 0.0, 1.0, 1.0));
    (*stateset)->addUniform(sunDirUniform);

    Uniform* suncolorUniform = new Uniform("suncolor", Vec4(1.0, 1.0, 1.0, 1.0));
    (*stateset)->addUniform(suncolorUniform);

    Uniform* skycolorUniform = new Uniform("skycolor", Vec4(0.5, 0.5, 1.0, 1.0));
    (*stateset)->addUniform(skycolorUniform);

    Uniform* skyfadingcolorUniform = new Uniform("skyfadingcolor", Vec4(0.8, 0.8, 0.8, 1.0));
    (*stateset)->addUniform(skyfadingcolorUniform);

    Uniform* skymaskingcolorUniform = new Uniform("skymaskingcolor", Vec4(1.0, 1.0, 1.0, 1.0));
    (*stateset)->addUniform(skymaskingcolorUniform);

    Uniform *matShaderToWorldUniform = new Uniform("shaderToWorldMat", Matrixd());
    (*stateset)->addUniform(matShaderToWorldUniform);

    Image* imageSky = osgDB::readImageFile(ANIMDataDir() + "Textures/NightSky.JPG");
    Texture2D* textureSky = new Texture2D(imageSky);    
    (*stateset)->setTextureAttributeAndModes(0, textureSky, StateAttribute::ON);

    Image* imagePara = osgDB::readImageFile(ANIMDataDir() + "Textures/Paramounts/Paramount00.JPG");
    Texture2D* texturePara = new Texture2D(imagePara);

    (*stateset)->setTextureAttributeAndModes(1, texturePara, StateAttribute::ON);

    Uniform* skyNightSampler = new Uniform("texNightSky", 0);
    (*stateset)->addUniform(skyNightSampler);

    Uniform* paraImageTextureSampler = new Uniform("texParamount", 1);
    (*stateset)->addUniform(paraImageTextureSampler);

    Program* programSky = new Program;
    (*stateset)->setAttribute(programSky);
    programSky->addShader(Shader::readShaderFile(Shader::VERTEX, ANIMDataDir() + "Shaders/EnvSky.vert"));
    programSky->addShader(Shader::readShaderFile(Shader::FRAGMENT, ANIMDataDir() + "Shaders/EnvSky.frag"));

    return skyDomeTrans;
}

};

