/***************************************************************
* Animation File Name: ANIMVirtualEarth.cpp
*
* Description: Create animated model of virtual sphere
*
* Written by ZHANG Lelin on Oct 7, 2010
*
***************************************************************/
#include "ANIMVirtualEarth.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMLoadVirtualEarthReferenceLevel()
***************************************************************/
void ANIMLoadVirtualEarthReferenceLevel(osg::Switch **designStateSwitch, osg::Geode **seasonsMapGeode)
{
    /* create seasons map geode for intersection test */
    *seasonsMapGeode = new Geode();
    Sphere* seasonsMapSphere = new Sphere();
    seasonsMapSphere->setRadius(ANIM_VIRTUAL_SEASONS_MAP_RADIUS);
    Drawable* seasonsMapDrawable = new ShapeDrawable(seasonsMapSphere);
    (*seasonsMapGeode)->addDrawable(seasonsMapDrawable);

    Material* material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setAlpha(Material::FRONT_AND_BACK, ANIM_VIRTUAL_SEASONS_MAP_ALPHA);

    Image* imgSeasonsMap = osgDB::readImageFile(ANIMDataDir() + "Textures/SeasonsMap.JPG");
    Texture2D* texSeasonsMap = new Texture2D(imgSeasonsMap);

    StateSet* stateset = new StateSet(); 
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON); 
    stateset->setTextureAttributeAndModes(0, texSeasonsMap, StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setMode(GL_LIGHTING, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::OPAQUE_BIN);
    (*seasonsMapGeode)->setStateSet(stateset);

    /* rotate seasons map to align March 21 to viewer's front */
    MatrixTransform *seasonsMapMatrixTrans = new MatrixTransform;
    Matrixf rotMat;
    rotMat.makeRotate(M_PI / 2, Vec3(0, 0, 1));
    seasonsMapMatrixTrans->setMatrix(rotMat);
    seasonsMapMatrixTrans->addChild(*seasonsMapGeode);

    /* load ecliptic ruler from VRML */
    Node* eclipticRulerNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/EclipticRuler.WRL");

    (*designStateSwitch)->addChild(seasonsMapMatrixTrans);
    (*designStateSwitch)->addChild(eclipticRulerNode);
}


/***************************************************************
* Function: ANIMLoadVirtualEarthEclipticLevel()
***************************************************************/
void ANIMLoadVirtualEarthEclipticLevel(osg::Switch **eclipticSwitch)
{
    Geode* wiredEclipticGeode = ANIMCreateWiredSphereGeode(12, 24, ANIM_VIRTUAL_SPHERE_RADIUS * 1.3f, Vec4(0, 1, 0, 1));
    Node* dateIndicatorNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/SeasonIndicator.WRL");
    Node* earthAxisNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/EarthAxis.WRL");
    Node* equatorRulerNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/EquatorRuler.WRL");

    /* rotate earth axis and equator ruler by 23.5 degrees from z-axis */
    Matrixd rotMat;
    float angle = 23.5f / 180.0f * M_PI;
    rotMat.makeRotate(Vec3(0, 0, 1), Vec3(sin(angle), 0, cos(angle)));

    MatrixTransform *earthAxisTrans = new MatrixTransform();
    MatrixTransform *equatorRulerTrans = new MatrixTransform();
    earthAxisTrans->setMatrix(rotMat);
    equatorRulerTrans->setMatrix(rotMat);

    earthAxisTrans->addChild(earthAxisNode);
    equatorRulerTrans->addChild(equatorRulerNode);
    (*eclipticSwitch)->addChild(wiredEclipticGeode);
    (*eclipticSwitch)->addChild(dateIndicatorNode);
    (*eclipticSwitch)->addChild(earthAxisTrans);
    (*eclipticSwitch)->addChild(equatorRulerTrans);
}


/***************************************************************
* Function: ANIMLoadVirtualEarthEquatorLevel()
***************************************************************/
void ANIMLoadVirtualEarthEquatorLevel(osg::Switch **equatorSwitch, osg::Geode **earthGeode,
		osg::PositionAttitudeTransform **xformTransFwd, osg::PositionAttitudeTransform **xformTransBwd, 
		osg::MatrixTransform **fixedPinIndicatorTrans, osg::MatrixTransform **fixedPinTrans)
{
    /* Child #0: load wiredEquatorGeode */
    Geode* wiredEquatorGeode = ANIMCreateWiredSphereGeode(12, 24, ANIM_VIRTUAL_SPHERE_RADIUS * 1.1f, Vec4(0, 1, 0, 1));

    /* Child #1, #2: forward / backward earth animation */
    *xformTransFwd = new PositionAttitudeTransform();
    *xformTransBwd = new PositionAttitudeTransform();
    *earthGeode = new Geode();
    Sphere* virtualEarth = new Sphere();
    Drawable* veDrawable = new ShapeDrawable(virtualEarth);
    
    virtualEarth->setRadius(ANIM_VIRTUAL_SPHERE_RADIUS);
    (*earthGeode)->addDrawable(veDrawable);
    (*xformTransFwd)->addChild(*earthGeode);
    (*xformTransBwd)->addChild(*earthGeode);

    /* set up the forward / backward scale animation path */
    AnimationPath* animationPathScaleFwd = new AnimationPath;
    AnimationPath* animationPathScaleBwd = new AnimationPath;
    animationPathScaleFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationPathScaleBwd->setLoopMode(AnimationPath::NO_LOOPING);
   
    Vec3 scaleFwd, scaleBwd;
    float step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
    for (int i = 0; i < ANIM_VIRTUAL_SPHERE_NUM_SAMPS + 1; i++)
    {
	float val = i * step;
	scaleFwd = Vec3(val, val, val);
	scaleBwd = Vec3(1-val, 1-val, 1-val);
	animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(Vec3(),Quat(), scaleFwd));
	animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(Vec3(),Quat(), scaleBwd));
    }

    AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
    (*xformTransFwd)->setUpdateCallback(animCallbackFwd);
    (*xformTransBwd)->setUpdateCallback(animCallbackBwd);

    /* apply shaders to geode stateset */
    StateSet* stateset = new StateSet();
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    (*earthGeode)->setStateSet(stateset);

    Program* shaderProg = new Program;
    stateset->setAttribute(shaderProg);
    shaderProg->addShader(Shader::readShaderFile(Shader::VERTEX, ANIMDataDir() + "Shaders/VirtualEarth.vert"));
    shaderProg->addShader(Shader::readShaderFile(Shader::FRAGMENT, ANIMDataDir() + "Shaders/VirtualEarth.frag"));

    Image* imgEarthDay = osgDB::readImageFile(ANIMDataDir() + "Textures/EarthDay.JPG");
    Texture2D* texEarthDay = new Texture2D(imgEarthDay);    
    stateset->setTextureAttributeAndModes(0, texEarthDay, StateAttribute::ON);

    Image* imgEarthNight = osgDB::readImageFile(ANIMDataDir() + "Textures/EarthNight.JPG");
    Texture2D* texEarthNight = new Texture2D(imgEarthNight);    
    stateset->setTextureAttributeAndModes(1, texEarthNight, StateAttribute::ON);

    Image* imgEarthClouds = osgDB::readImageFile(ANIMDataDir() + "Textures/EarthClouds.JPG");
    Texture2D* texEarthClouds = new Texture2D(imgEarthClouds);    
    stateset->setTextureAttributeAndModes(2, texEarthClouds, StateAttribute::ON);

    Uniform* earthDaySampler = new Uniform("EarthDay", 0);
    stateset->addUniform(earthDaySampler);

    Uniform* earthNightSampler = new Uniform("EarthNight", 1);
    stateset->addUniform(earthNightSampler);

    Uniform* earthCloudsSampler = new Uniform("EarthCloudGloss", 2);
    stateset->addUniform(earthCloudsSampler);

    Uniform* lightPosUniform = new Uniform("LightPos", Vec4(ANIMVirtualEarthLightDir(), 0));
    stateset->addUniform(lightPosUniform);

    /* Child #3 #4 #5: pin indicator trans, fixed pin trans */
    *fixedPinIndicatorTrans = new MatrixTransform();
    Node* pinIndicatorNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/TimeIndicator.WRL");
    (*fixedPinIndicatorTrans)->addChild(pinIndicatorNode);

    *fixedPinTrans = new MatrixTransform();
    Node* pinHeadNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/PinHead.WRL");
    Node* pinButtNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/PinButt.WRL");
    (*fixedPinTrans)->addChild(pinHeadNode);
    (*fixedPinTrans)->addChild(pinButtNode);

    /* Attach all children to scene graph on equator level */
    (*equatorSwitch)->addChild(wiredEquatorGeode);
    (*equatorSwitch)->addChild(*xformTransFwd);
    (*equatorSwitch)->addChild(*xformTransBwd);
    (*equatorSwitch)->addChild(*fixedPinIndicatorTrans);
    (*equatorSwitch)->addChild(*fixedPinTrans);
}


/***************************************************************
* Function: ANIMVirtualEarthLightDir()
***************************************************************/
const Vec3 ANIMVirtualEarthLightDir() { return Vec3(1, 0, 0); }


/***************************************************************
* Function: ANIMCreateWiredSphereGeode()
***************************************************************/
Geode *ANIMCreateWiredSphereGeode(const int numLatiSegs, const int numLongiSegs, const float rad, const osg::Vec4 color)
{
    Geode *wiredSphereGeode = new Geode;
    Geometry* wiredSphereGeometry = new Geometry();
    wiredSphereGeode->addDrawable(wiredSphereGeometry);
    Vec3Array* vertices = new Vec3Array;

    /* setup coordinates of wire frame geometries */
    const int numVerts = (numLatiSegs - 1) * numLongiSegs + 1;
    const float phiStep = M_PI / numLatiSegs;
    const float thetaStep = M_PI * 2 / numLongiSegs;
    float phi = 0, theta = 0, cx, cy, cz;
    vertices->push_back(Vec3(0, 0, rad));
    for (int i = 1; i < numLatiSegs; i++)
    {
	theta = thetaStep * i;
	for (int j = 0; j < numLongiSegs; j++)
	{
	    phi = j * phiStep;
	    cx = rad * sin(theta) * cos(phi);
	    cy = rad * sin(theta) * sin(phi);
	    cz = rad * cos(theta);
	    vertices->push_back(Vec3(cx, cy, cz));
	}
    }
    vertices->push_back(Vec3(0, 0, -rad));
    wiredSphereGeometry->setVertexArray(vertices);

    /* setup connections between adjacent vertices */
    for (int i = 1; i < numLatiSegs; i++)		// draw latitudes
    {
	DrawElementsUInt* edges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
	int idxStart = (i - 1) * numLongiSegs + 1;
	int idxEnd = idxStart + numLongiSegs - 1;
	for (int k = idxStart; k <= idxEnd; k++) edges->push_back(k);
	edges->push_back(idxStart);
	wiredSphereGeometry->addPrimitiveSet(edges);
    }
    for (int j = 0; j < numLongiSegs; j++)		// draw longitudes
    {
	DrawElementsUInt* edges = new DrawElementsUInt(PrimitiveSet::LINE_STRIP, 0);
	int idxStart = j + 1;
	edges->push_back(0);
	for (int k = 0; k < numLatiSegs - 1; k++) edges->push_back(idxStart + k * numLongiSegs);
	edges->push_back(numVerts - 1);
	wiredSphereGeometry->addPrimitiveSet(edges);
    } 

    /* apply color stateset to wire frames */
    StateSet* stateset = new StateSet();
    Material* material = new Material;
    material->setAmbient(Material::FRONT_AND_BACK, color);
    material->setDiffuse(Material::FRONT_AND_BACK, color);
    material->setAlpha(Material::FRONT_AND_BACK, 0.25f);
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setMode(GL_LIGHTING, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::OPAQUE_BIN);
    wiredSphereGeode->setStateSet(stateset);

    return wiredSphereGeode;
}



};














