/***************************************************************
* Animation File Name: ANIMRefWaterSurf.cpp
*
* Description: Create reference water surface
*
* Written by ZHANG Lelin on Oct 22, 2010
*
***************************************************************/
#include "ANIMRefWaterSurf.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMCreateRefWaterSurf()
*
***************************************************************/
MatrixTransform *ANIMCreateRefWaterSurf(StateSet **stateset, Geometry **watersurfGeometry)
{
    MatrixTransform *WaterSurfTrans = new MatrixTransform;
    Matrixd transmat;
    transmat.makeTranslate(Vec3(0, 0, ANIM_WATERSURF_ALTITUDE));
    WaterSurfTrans->setMatrix(transmat);

    /* reflective water surface */
    Geode *watersurfGeode = new Geode();
    (*watersurfGeometry) = new Geometry();

    Vec3Array* vertices = new Vec3Array;
    Vec3Array* normals = new Vec3Array;
    Vec2Array* texcoords0 = new Vec2Array(4);
    Vec2Array* texcoords2 = new Vec2Array(4);

    vertices->push_back(Vec3(ANIM_WATERSURF_SIZE, ANIM_WATERSURF_SIZE, 0));
    vertices->push_back(Vec3(-ANIM_WATERSURF_SIZE, ANIM_WATERSURF_SIZE, 0));
    vertices->push_back(Vec3(-ANIM_WATERSURF_SIZE, -ANIM_WATERSURF_SIZE, 0));
    vertices->push_back(Vec3(ANIM_WATERSURF_SIZE, -ANIM_WATERSURF_SIZE, 0));

    (*texcoords0)[0].set(ANIM_WATERSURF_SIZE / 7, ANIM_WATERSURF_SIZE / 7);
    (*texcoords0)[1].set(0.0f, 0.0f);
    (*texcoords0)[2].set(0.0f, ANIM_WATERSURF_SIZE / 7);
    (*texcoords0)[3].set(ANIM_WATERSURF_SIZE / 7, 0.0f);
    (*texcoords2)[0] = (*texcoords0)[0];
    (*texcoords2)[1] = (*texcoords0)[1];
    (*texcoords2)[2] = (*texcoords0)[2];
    (*texcoords2)[3] = (*texcoords0)[3];
    for (int i = 0; i < 4; i++) normals->push_back(Vec3(0, 0, 1));

    DrawElementsUInt* square = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);
    square->push_back(0);	square->push_back(1);
    square->push_back(2);	square->push_back(3);

    (*watersurfGeometry)->addPrimitiveSet(square);
    (*watersurfGeometry)->setVertexArray(vertices);
    (*watersurfGeometry)->setNormalArray(normals);
    (*watersurfGeometry)->setNormalBinding(Geometry::BIND_PER_VERTEX);
    (*watersurfGeometry)->setTexCoordArray(0, texcoords0);
    (*watersurfGeometry)->setTexCoordArray(2, texcoords2);

    watersurfGeode->addDrawable(*watersurfGeometry);
    WaterSurfTrans->addChild(watersurfGeode);

    (*stateset) = new StateSet();
    watersurfGeode->setStateSet(*stateset);

    /* reflective water surface shader */
    Image* imageNormmap = osgDB::readImageFile(ANIMDataDir() + "Textures/NormalMap.BMP");
    Texture2D* textureNormmap = new Texture2D(imageNormmap); 
    textureNormmap->setWrap(Texture::WRAP_S,Texture::MIRROR);
    textureNormmap->setWrap(Texture::WRAP_T,Texture::MIRROR);

    Image* imagePara = osgDB::readImageFile(ANIMDataDir() + "Textures/Paramounts/Paramount00.JPG");
    Texture2D* texturePara = new Texture2D(imagePara);

    Image* imagePebbleBase = osgDB::readImageFile(ANIMDataDir() + "Textures/PebbleBase.JPG");
    Texture2D* textureBase = new Texture2D(imagePebbleBase); 
    textureBase->setWrap(Texture::WRAP_S,Texture::MIRROR);
    textureBase->setWrap(Texture::WRAP_T,Texture::MIRROR);

    (*stateset)->setTextureAttributeAndModes(0, textureNormmap, StateAttribute::ON);
    (*stateset)->setTextureAttributeAndModes(1, texturePara, StateAttribute::ON);
    (*stateset)->setTextureAttributeAndModes(2, textureBase, StateAttribute::ON);

    Uniform* skyRadiusUniform = new Uniform("skyradius", ANIM_SKYDOME_RADIUS);
    (*stateset)->addUniform(skyRadiusUniform);

    Uniform* basetexSizeUniform = new Uniform("basetexsize", 
	ANIM_WATERSURF_SIZE * 2.0f / (ANIM_WATERSURF_SIZE / 7.0f));
    (*stateset)->addUniform(basetexSizeUniform);

    Uniform* viewposUniform = new Uniform("viewpos", Vec3(0.0, 0.0, 0.0));
    (*stateset)->addUniform(viewposUniform);

    Uniform* sundirUniform = new Uniform("sundir", Vec3(0.0, 0.0, 1.0));
    (*stateset)->addUniform(sundirUniform);

    Uniform* suncolorUniform = new Uniform("suncolor", Vec4(1.0, 1.0, 1.0, 1.0));
    (*stateset)->addUniform(suncolorUniform);

    Uniform* skycolorUniform = new Uniform("skycolor", Vec4(0.5, 0.5, 1.0, 1.0));
    (*stateset)->addUniform(skycolorUniform);

    Uniform* watermaskingcolorUniform = new Uniform("watermaskingcolor", Vec4(1.0, 1.0, 1.0, 1.0));
    (*stateset)->addUniform(watermaskingcolorUniform);

    Uniform* normmapSampler = new Uniform("texNormalMap", 0);
    (*stateset)->addUniform(normmapSampler);

    Uniform* paraImageTextureSampler = new Uniform("texParaImage", 1);
    (*stateset)->addUniform(paraImageTextureSampler);

    Uniform* pebbleBaseSampler = new Uniform("texPebbleBase", 2);
    (*stateset)->addUniform(pebbleBaseSampler);

    Program* programWatersurf = new Program;
    (*stateset)->setAttribute(programWatersurf);
    programWatersurf->addShader(Shader::readShaderFile
				(Shader::VERTEX, ANIMDataDir() + "Shaders/EnvWatersurf.vert"));
    programWatersurf->addShader(Shader::readShaderFile
				(Shader::FRAGMENT, ANIMDataDir() + "Shaders/EnvWatersurf.frag"));

    return WaterSurfTrans;
}

};








