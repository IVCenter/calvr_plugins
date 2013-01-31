/***************************************************************
* File Name: VirtualScenicHandler.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Oct 22, 2010
*
***************************************************************/

#include "VirtualScenicHandler.h"
#include <cvrKernel/PluginHelper.h>

using namespace std;
using namespace osg;


// Constructor
VirtualScenicHandler::VirtualScenicHandler(Group* nonIntersectableSceneGraphPtr,    
    osg::Group* intersectableSceneGraphPtr): mFloorplanIdx(-1)
{
    intersectableSceneGraphPtr->addChild(createSunLight(intersectableSceneGraphPtr->getOrCreateStateSet()));
    nonIntersectableSceneGraphPtr->addChild(createPointLight(nonIntersectableSceneGraphPtr->getOrCreateStateSet()));

    mWaterEnabled = false;

    // load reference plane and sky dome
    mXYPlaneSwitch = new Switch();
    mSkyDomeSwitch = new Switch();
    mFloorplanSwitch = new Switch();

    mXYPlaneSwitch->addChild(CAVEAnimationModeler::ANIMCreateRefXYPlane());
    mSkyDomeSwitch->addChild(CAVEAnimationModeler::ANIMCreateRefSkyDome(&mSkyDomeStateset));

    intersectableSceneGraphPtr->addChild(mXYPlaneSwitch);
    nonIntersectableSceneGraphPtr->addChild(mSkyDomeSwitch);
    intersectableSceneGraphPtr->addChild(mFloorplanSwitch);

    mXYPlaneSwitch->setAllChildrenOff();
    mSkyDomeSwitch->setAllChildrenOff();
    mFloorplanSwitch->setAllChildrenOff();
    
    mWaterSurfSwitch = NULL;
    if (mWaterEnabled)
    {
        mWaterSurfSwitch = new Switch();
        mWaterSurfSwitch->addChild(CAVEAnimationModeler::ANIMCreateRefWaterSurf(&mWatersurfStateset, &mWatersurfGeometry));
        intersectableSceneGraphPtr->addChild(mWaterSurfSwitch);
        mWaterSurfSwitch->setAllChildrenOff();
    }
}


/***************************************************************
* Function: setGeometryVisible()
***************************************************************/
void VirtualScenicHandler::setGeometryVisible(bool flag)
{
    if (flag) 
    {
        mXYPlaneSwitch->setAllChildrenOn();
        mSkyDomeSwitch->setAllChildrenOn();
        if (mWaterEnabled)
        {
            mWaterSurfSwitch->setAllChildrenOn();
        }
    }
    else 
    {
        mXYPlaneSwitch->setAllChildrenOff();
        mSkyDomeSwitch->setAllChildrenOff();
        if (mWaterEnabled)
        {
            mWaterSurfSwitch->setAllChildrenOff();
        }
        mFloorplanSwitch->setAllChildrenOff();
    }
}


/***************************************************************
* Function: setSkyMaskingColorEnabled()
***************************************************************/
void VirtualScenicHandler::setSkyMaskingColorEnabled(bool flag)
{
    Uniform *skymaskingcolorUniform = mSkyDomeStateset->getOrCreateUniform("skymaskingcolor", Uniform::FLOAT_VEC4);
    Uniform *watermaskingcolorUniform;
    if (mWaterEnabled)
    {
         watermaskingcolorUniform = 
            mWatersurfStateset->getOrCreateUniform("watermaskingcolor", Uniform::FLOAT_VEC4);
    }

    if (flag)
    {
        skymaskingcolorUniform->set(Vec4(0.5, 0.8, 0.5, 1.0));
        if (mWaterEnabled)
            watermaskingcolorUniform->set(Vec4(0.5, 0.8, 0.5, 1.0));
    }
    else 
    {
        skymaskingcolorUniform->set(Vec4(1.0, 1.0, 1.0, 1.0));
        if (mWaterEnabled)
            watermaskingcolorUniform->set(Vec4(1.0, 1.0, 1.0, 1.0));
    }
}


/***************************************************************
* Function: setVSParamountPreviewHighlight()
***************************************************************/
void VirtualScenicHandler::setVSParamountPreviewHighlight(bool flag, Geode *paintGeode)
{
    StateSet *stateset = paintGeode->getOrCreateStateSet();
    Material *material = dynamic_cast<Material*> (stateset->getAttribute(StateAttribute::MATERIAL)); 
    if (!material) 
        material = new Material;
    if (flag)
    {
        material->setAlpha(Material::FRONT_AND_BACK, 0.5f);
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.3, 1.0, 0.3, 1.0));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.3, 1.0, 0.3, 1.0));
    }
    else
    {
        material->setAlpha(Material::FRONT_AND_BACK, 1.0f);
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    }
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
}


/***************************************************************
* Function: setFloorplanPreviewHighlight()
***************************************************************/
void VirtualScenicHandler::setFloorplanPreviewHighlight(bool flag, Geode *pageGeode)
{
    StateSet *stateset = pageGeode->getOrCreateStateSet();
    Material *material = dynamic_cast<Material*> (stateset->getAttribute(StateAttribute::MATERIAL)); 
    if (!material)  
    {
        material = new Material;
    }

    if (flag)
    {
        material->setAlpha(Material::FRONT_AND_BACK, 0.5f);
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.8, 0.4, 0.0, 1.0));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.8, 0.4, 0.0, 1.0));
    }
    else
    {
        material->setAlpha(Material::FRONT_AND_BACK, 1.0f);
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    }
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
}


/***************************************************************
* Function: switchVSParamount()
***************************************************************/
void VirtualScenicHandler::switchVSParamount(const std::string &texname)
{
    Image* imagePara = osgDB::readImageFile(texname);
    Texture2D* texturePara = new Texture2D(imagePara);
    mSkyDomeStateset->setTextureAttributeAndModes(1, texturePara, StateAttribute::ON);
    if (mWaterEnabled)
        mWatersurfStateset->setTextureAttributeAndModes(1, texturePara, StateAttribute::ON);
}


/***************************************************************
* Function: updateVSParameters()
*
* Description: This function is called by 'DSVirtualEarth' under
* 'DesignStateRenderer'. Input parameter 'transMat' is the matrix
* that transformed vector in shader space to world space.
*
***************************************************************/
void VirtualScenicHandler::updateVSParameters(const Matrixd &matShaderToWorld,
		const Vec3 &sunDirWorld, const Vec3 &viewDir, const Vec3 &viewPos)
{
    Matrixd matWorldToShader = Matrixd::inverse(matShaderToWorld);

    /*  'sunDirWorld' is directional vector that illuminates the virtual earth sphere, 
	'sunDirShader' is also defined in world space and represents for the actual 
	 sun direction of the environment.
    */
    Vec3 sunDirShader = sunDirWorld * matWorldToShader;
    sunDirShader.normalize();

    float haze = viewDir * sunDirShader;	haze = haze > 0 ? haze:0;
    float hazeRadisuMin = 0.975f - sunDirShader.z() * 0.02f - haze * 0.1f;
    float hazeRadisuMax = 0.995f;

    Vec4 sunColor, skyColor, skyFadingColor;
    interpretColorParameters(sunDirShader, sunColor, skyColor, skyFadingColor);

    /* Update light sources of scene graphs under DesignObjectHandler:
       Set relatively high ambient components for 'mSunLight' so that geometries are
       still visible at night time. For 'mPointLight', always keep its position on top
       of viewer's head so all reference geometries are illuminated evenly.
    */
    mSunLight->setPosition(Vec4(sunDirShader, 0.0f));
    mSunLight->setDiffuse(sunColor);
    mSunLight->setAmbient(Vec4(0.5f,0.5f,0.5f,1.0f));

    mPointLight->setPosition(Vec4(viewPos, 0.0f));

    // update sky dome uniform list
    Uniform *hazeRadisuMinUniform = mSkyDomeStateset->getOrCreateUniform("hazeRadisuMin", Uniform::FLOAT);
    hazeRadisuMinUniform->set(hazeRadisuMin);

    Uniform *hazeRadisuMaxUniform = mSkyDomeStateset->getOrCreateUniform("hazeRadisuMax", Uniform::FLOAT);
    hazeRadisuMaxUniform->set(hazeRadisuMax);

    Uniform *sunDirUniform1 = mSkyDomeStateset->getOrCreateUniform("sundir", Uniform::FLOAT_VEC4);
    sunDirUniform1->set(Vec4(sunDirShader, 0.0));

    Uniform *suncolorUniform1 = mSkyDomeStateset->getOrCreateUniform("suncolor", Uniform::FLOAT_VEC4);
    suncolorUniform1->set(sunColor);

    Uniform *skycolorUniform1 = mSkyDomeStateset->getOrCreateUniform("skycolor", Uniform::FLOAT_VEC4);
    skycolorUniform1->set(skyColor);

    Uniform *skyfadingcolorUniform = mSkyDomeStateset->getOrCreateUniform("skyfadingcolor", Uniform::FLOAT_VEC4);
    skyfadingcolorUniform->set(skyFadingColor);

    Uniform *matShaderToWorldUniform = mSkyDomeStateset->getOrCreateUniform("shaderToWorldMat", Uniform::FLOAT_MAT4);
    matShaderToWorldUniform->set(matWorldToShader);

    // update water surface uniform list
    if (mWaterEnabled)
    {
        Uniform *sunDirUniform2 = mWatersurfStateset->getOrCreateUniform("sundir", Uniform::FLOAT_VEC3);
        sunDirUniform2->set(sunDirShader);

        Uniform *suncolorUniform2 = mWatersurfStateset->getOrCreateUniform("suncolor", Uniform::FLOAT_VEC4);
        suncolorUniform2->set(sunColor);
    }
    Uniform *skycolorUniform2 = mSkyDomeStateset->getOrCreateUniform("skycolor", Uniform::FLOAT_VEC4);
    skycolorUniform2->set(skyColor);

    // change texture coordinates of normal map
    if (mWaterEnabled)
    {
        Array* texcoordArray = mWatersurfGeometry->getTexCoordArray(0);
        if (texcoordArray->getType() == Array::Vec2ArrayType)
        {
            Vec2* texDataPtr = (Vec2*)(texcoordArray->getDataPointer());
            float randArr[2];
            for (int i = 0; i < 2; i++) 
                randArr[i] = (float)rand() / RAND_MAX * ANIM_WATERSURF_TEXCOORD_NOISE_LEVEL;
            texDataPtr[0] += Vec2(randArr[0], randArr[1]);
            texDataPtr[1] += Vec2(randArr[0], randArr[1]);
            texDataPtr[2] += Vec2(randArr[0], randArr[1]);
            texDataPtr[3] += Vec2(randArr[0], randArr[1]);
        }

        mWatersurfGeometry->dirtyDisplayList();
        mWatersurfGeometry->dirtyBound();
    }
}


/***************************************************************
* Function: switchFloorplan()
***************************************************************/
void VirtualScenicHandler::switchFloorplan(const int &idx, const VisibilityOption &option)
{
    if (idx >= mFloorplanSwitch->getNumChildren()) 
    {
        return;
    }

    if (option == INVISIBLE)
    {
        if (mFloorplanIdx >= 0) 
        {
            mFloorplanSwitch->setSingleChildOn(mFloorplanIdx);
        }
        else 
        {
            mFloorplanSwitch->setAllChildrenOff();
        }
    }
    else 
    {
        Geode *floorplanGeode = dynamic_cast <Geode*> (mFloorplanSwitch->getChild(idx));
        StateSet *stateset = floorplanGeode->getOrCreateStateSet();
        Material *material = dynamic_cast<Material*> (stateset->getAttribute(StateAttribute::MATERIAL)); 

        if (!material) 
        {
            material = new Material;
        }

        if (option == TRANSPARENT)
        {
            material->setAlpha(Material::FRONT_AND_BACK, 0.5f);
        }
        else if (option == SOLID)
        {
            material->setAlpha(Material::FRONT_AND_BACK, 1.0f);
            mFloorplanIdx = idx;
        }
        mFloorplanSwitch->setSingleChildOn(idx);
    }
}


/***************************************************************
* Function: createFloorplanGeometry()
***************************************************************/
void VirtualScenicHandler::createFloorplanGeometry(const int numPages, 
    CAVEAnimationModeler::ANIMPageEntry **pageEntryArray)
{
    if (numPages <= 0)
    {
        return;
    }

    for (int i = 0; i < numPages; i++)
    {
        // create floorplan geometry
        float length = pageEntryArray[i]->mLength;
        float width = pageEntryArray[i]->mWidth;
        float altitude = pageEntryArray[i]->mAlti;

        Geode *floorplanGeode = new Geode;
        Geometry *floorplanGeometry = new Geometry;

        Vec3Array* vertices = new Vec3Array;
        Vec3Array* normals = new Vec3Array;
        Vec2Array* texcoords = new Vec2Array(4);

        vertices->push_back(Vec3(-length / 2,  width / 2, altitude));	(*texcoords)[0].set(0, 1);
        vertices->push_back(Vec3(-length / 2, -width / 2, altitude));	(*texcoords)[1].set(0, 0);
        vertices->push_back(Vec3( length / 2, -width / 2, altitude));	(*texcoords)[2].set(1, 0);
        vertices->push_back(Vec3( length / 2,  width / 2, altitude));	(*texcoords)[3].set(1, 1);

        for (int k = 0; k < 4; k++) 
        { 
            normals->push_back(Vec3(0, 0, 1));
        }

        DrawElementsUInt* rectangle = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);
        rectangle->push_back(0);	rectangle->push_back(1);
        rectangle->push_back(2);	rectangle->push_back(3);

        floorplanGeometry->addPrimitiveSet(rectangle);
        floorplanGeometry->setVertexArray(vertices);
        floorplanGeometry->setNormalArray(normals);
        floorplanGeometry->setTexCoordArray(0, texcoords);
        floorplanGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);

        floorplanGeode->addDrawable(floorplanGeometry);
        mFloorplanSwitch->addChild(floorplanGeode);

        /* load floorplan images */
        Material *transmaterial = new Material;
        transmaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
        transmaterial->setAlpha(Material::FRONT_AND_BACK, 1.0f);

        Image* imgFloorplan = osgDB::readImageFile(pageEntryArray[i]->mTexFilename);

        Texture2D* texFloorplan = new Texture2D(imgFloorplan); 

        StateSet *floorplanStateSet = floorplanGeode->getOrCreateStateSet();
        floorplanStateSet->setTextureAttributeAndModes(0, texFloorplan, StateAttribute::ON);
        floorplanStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
        floorplanStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
        floorplanStateSet->setAttributeAndModes(transmaterial, StateAttribute::OVERRIDE | StateAttribute::ON);
    }
}


/***************************************************************
* Function: createSunLight()
***************************************************************/
Group *VirtualScenicHandler::createSunLight(osg::StateSet *stateset)
{
    Group* lightGroup = new Group;
    mSunLightSource = new LightSource; 
    mSunLight = new Light;

    mSunLight->setLightNum(4);
    mSunLight->setPosition(Vec4(0.0f, 1.0f, 1.0f, 0.0f));
    mSunLight->setDiffuse(Vec4(1.0f,1.0f,1.0f,1.0f));
    mSunLight->setSpecular(Vec4(0.0f,0.0f,0.0f,1.0f));
    mSunLight->setAmbient(Vec4(0.5f,0.5f,0.5f,1.0f));
  
    mSunLightSource->setLight(mSunLight);
    mSunLightSource->setLocalStateSetModes(StateAttribute::ON);
    mSunLightSource->setStateSetModes(*stateset, StateAttribute::ON);

    lightGroup->addChild(mSunLightSource);

    return lightGroup;
}


/***************************************************************
* Function: createPointLight()
***************************************************************/
Group *VirtualScenicHandler::createPointLight(osg::StateSet *stateset)
{
    Group* lightGroup = new Group;
    mPointLightSource = new LightSource; 
    mPointLight = new Light;

    mPointLight->setLightNum(5);
    mPointLight->setPosition(Vec4(0.0f, 0.0f, -0.5f, 1.0f));
    mPointLight->setDiffuse(Vec4(1.0f,1.0f,1.0f,1.0f));
    mPointLight->setSpecular(Vec4(0.0f,0.0f,0.0f,1.0f));
    mPointLight->setAmbient(Vec4(0.2f,0.2f,0.2f,1.0f));
  
    mPointLightSource->setLight(mPointLight);
    mPointLightSource->setLocalStateSetModes(StateAttribute::ON);
    mPointLightSource->setStateSetModes(*stateset, StateAttribute::ON);

    lightGroup->addChild(mPointLightSource);

    return lightGroup;
}


/***************************************************************
* Function: interpretColorParameters()
***************************************************************/
void VirtualScenicHandler::interpretColorParameters(const Vec3 &sunDir, Vec4 &sunColor, 
						Vec4 &skyColor, Vec4 &skyFadingColor)
{
    float interp;
    float sunHeight = sunDir.z();

    if (sunHeight > -0.2) 
    {
        float r, g, b;
        if (sunHeight > 0.25) 
        { 
            r = 1;  g = 1;   b = 1; 
        }
        else if (sunHeight > 0.0)
        {
            r = 1.0f;		r = r > 0 ? r:0;
            g = sunHeight * 3.2 + 0.2;	g = g > 0 ? g:0;
            b = sunHeight * 3.2 + 0.2;	b = b > 0 ? b:0;
            r = sqrt(r);    g = sqrt(g);    b = sqrt(b);
        }
        else 
        { 
            r = 1.0f;    g = 0.2f;    b = 0.2f; 
        }
        sunColor = Vec4(r, g, b, 1);
    } 
    else 
    {
        sunColor = Vec4(0, 0, 0, 1);
    }

    Vec4 skyColor1(0.0, 0.0, 0.0, 1.0), skyColor2(0.0, 0.0, 0.0, 1.0);
    if (sunHeight < -0.3)
    {
        interp = 0.0;
        skyFadingColor = Vec4(0, 0, 0, 1);
    }
    else if (sunHeight < -0.1) 
    { 
        interp = (sunHeight + 0.3)/0.2; 
        skyColor1 = Vec4(0.0, 0.0, 0.0, 1.0); 
        skyColor2 = Vec4(0.0, 0.0, 0.3, 1.0);
        skyFadingColor = skyColor1;
    }
    else if (sunHeight < 0.1) 
    { 
        interp = (sunHeight + 0.1)/0.1; 
        skyColor1 = Vec4(0.0, 0.0, 0.3, 1.0); 
        skyColor2 = Vec4(0.6, 0.0, 0.1, 1.0); 
        skyFadingColor = skyColor1;
    }
    else if (sunHeight < 0.2) 
    { 
        interp = (sunHeight - 0.1)/0.1; 
        skyColor1 = Vec4(0.6, 0.0, 0.1, 1.0); 
        skyColor2 = Vec4(0.8, 0.6, 0.1, 1.0);
        skyFadingColor = skyColor1;
    }
    else if (sunHeight < 0.4) 
    { 
        interp = (sunHeight - 0.1)/0.2; 
        skyColor1 = Vec4(0.8, 0.6, 0.1, 1.0); 
        skyColor2 = Vec4(0.5, 0.5, 1.0, 1.0);
        skyFadingColor = Vec4(0.8, 0.8, 0.8, 1.0);
    }
    else 
    { 
        interp = 1.0;
        skyColor1 = Vec4(0.5, 0.5, 1.0, 1.0); 
        skyColor2 = Vec4(0.5, 0.5, 1.0, 1.0); 
        skyFadingColor = Vec4(0.8, 0.8, 0.8, 1.0);
    }
    skyColor = skyColor1 * (1.0f - interp) + skyColor2 * interp;
}

