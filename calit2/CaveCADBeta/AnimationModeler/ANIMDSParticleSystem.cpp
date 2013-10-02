/***************************************************************
* Animation File Name: ANIMDSParticleSystem.cpp
*
* Description: Create design state particle system
*
* Written by ZHANG Lelin on Oct 5, 2010
*
***************************************************************/
#include "ANIMDSParticleSystem.h"

using namespace std;
using namespace osg;


namespace CAVEAnimationModeler
{

/***************************************************************
* Function: ANIMCreateDesignStateParticleSystem()
*
***************************************************************/
MatrixTransform* ANIMCreateDesignStateParticleSystem(ANIMEmitterList *dsEmitterList)
{
    MatrixTransform *psTrans = new MatrixTransform;
    MatrixTransform *psRedflameTrans = new MatrixTransform;
    MatrixTransform *psGreenflameTrans = new MatrixTransform;
    PositionAttitudeTransform* psRotTrans = new PositionAttitudeTransform;

    psRotTrans->addChild(psRedflameTrans);
    psRotTrans->addChild(psGreenflameTrans);
    psTrans->addChild(psRotTrans);

    /* create rotation animation callback for particle system */
    AnimationPath* animationPathRot = new AnimationPath;
    animationPathRot->setLoopMode(AnimationPath::LOOP);

    Quat rotation;
    float step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
    for (int i = 0; i < ANIM_VIRTUAL_SPHERE_NUM_SAMPS; i++)
    {
	float val = i * step;
	rotation = Quat(M_PI * 2 * val , Vec3(0, 0, 1));
	animationPathRot->insert(val, AnimationPath::ControlPoint(Vec3(), Quat(), Vec3(1, 1, 1)));
    }
    psRotTrans->setUpdateCallback(new AnimationPathCallback(animationPathRot, 0.0, 100.0));

    /* create green flame particle system */
    osgParticle::Particle ptemplate1;
    ptemplate1.setLifeTime(2);
    ptemplate1.setShape(osgParticle::Particle::QUAD);
    ptemplate1.setSizeRange(osgParticle::rangef(0.f, 48.f));
    ptemplate1.setAlphaRange(osgParticle::rangef(0.4f, 0.6f));
    ptemplate1.setColorRange(osgParticle::rangev4(Vec4(0.2, 0.2, 0.2, 1.0), Vec4(0.2, 0.2, 0.2, 1.0)));
    ptemplate1.setRadius(0.5f);
    ptemplate1.setMass(0.05f);

    osgParticle::ParticleSystem *ps1 = new osgParticle::ParticleSystem;
    ps1->setDefaultAttributes(ANIMDataDir() + "Textures/ParticleFireGreen.JPG", true, false);
    ps1->setDefaultParticleTemplate(ptemplate1);

    osgParticle::ModularEmitter *emitter1 = new osgParticle::ModularEmitter;
    
    osgParticle::RandomRateCounter *counter1 = new osgParticle::RandomRateCounter;
    counter1->setRateRange(800, 1200);
    emitter1->setCounter(counter1);

    osgParticle::SectorPlacer *placer1 = new osgParticle::SectorPlacer;
    placer1->setCenter(0, 0, -0.2);
    placer1->setRadiusRange(0.1, 0.28);
    placer1->setPhiRange(0, 2 * osg::PI);
    emitter1->setPlacer(placer1);

    osgParticle::RadialShooter *shooter1 = new osgParticle::RadialShooter;
    shooter1->setInitialSpeedRange(0, 0.1);
    emitter1->setShooter(shooter1);

    osgParticle::ModularProgram *program1 = new osgParticle::ModularProgram;

    osgParticle::AccelOperator *op1 = new osgParticle::AccelOperator;
    op1->setAcceleration(Vec3(0, 0, 0.0002));
    program1->addOperator(op1);

    Geode *geode1 = new osg::Geode;

    osgParticle::ParticleSystemUpdater *psu1 = new osgParticle::ParticleSystemUpdater;
 
    emitter1->setParticleSystem(ps1);
    program1->setParticleSystem(ps1);
    psu1->addParticleSystem(ps1);
    geode1->addDrawable(ps1);

    psGreenflameTrans->addChild(emitter1);
    psGreenflameTrans->addChild(program1);
    psGreenflameTrans->addChild(psu1);
    psGreenflameTrans->addChild(geode1);

    /* create red flame particle system */
    osgParticle::Particle ptemplate2;
    ptemplate2.setLifeTime(1.5);
    ptemplate2.setShape(osgParticle::Particle::QUAD);
    ptemplate2.setSizeRange(osgParticle::rangef(0.f, 24.f));
    ptemplate2.setAlphaRange(osgParticle::rangef(0.6f, 0.8f));
    ptemplate2.setColorRange(osgParticle::rangev4(Vec4(0.2, 0.2, 0.2, 1.0), Vec4(0.2, 0.2, 0.2, 1.0)));
    ptemplate2.setRadius(0.5f);
    ptemplate2.setMass(0.05f);

    osgParticle::ParticleSystem *ps2 = new osgParticle::ParticleSystem;
    ps2->setDefaultAttributes(ANIMDataDir() + "Textures/ParticleFireRed.JPG", true, false);
    ps2->setDefaultParticleTemplate(ptemplate2);

    osgParticle::ModularEmitter *emitter2 = new osgParticle::ModularEmitter;
    
    osgParticle::RandomRateCounter *counter2 = new osgParticle::RandomRateCounter;
    counter2->setRateRange(400, 600);
    emitter2->setCounter(counter2);

    osgParticle::SectorPlacer *placer2 = new osgParticle::SectorPlacer;
    placer2->setCenter(0, 0, -0.2);
    placer2->setRadiusRange(0.1, 0.28);
    placer2->setPhiRange(0, 2 * osg::PI);
    emitter2->setPlacer(placer2);

    osgParticle::RadialShooter *shooter2 = new osgParticle::RadialShooter;
    shooter2->setInitialSpeedRange(0, 0.4);
    emitter2->setShooter(shooter2);

    osgParticle::ModularProgram *program2 = new osgParticle::ModularProgram;

    osgParticle::AccelOperator *op2 = new osgParticle::AccelOperator;
    op2->setAcceleration(Vec3(0, 0, 0.0003));
    program2->addOperator(op2);

    Geode *geode2 = new osg::Geode;

    osgParticle::ParticleSystemUpdater *psu2 = new osgParticle::ParticleSystemUpdater;
 
    emitter2->setParticleSystem(ps2);
    program2->setParticleSystem(ps2);
    psu2->addParticleSystem(ps2);
    geode2->addDrawable(ps2);

    psRedflameTrans->addChild(emitter2);
    psRedflameTrans->addChild(program2);
    psRedflameTrans->addChild(psu2);
    psRedflameTrans->addChild(geode2);

    /* add emitters to design state's emitter control list */
    dsEmitterList->push_back(emitter1);
    dsEmitterList->push_back(emitter2);
    return psTrans;
}


/***************************************************************
* Function: ANIMCreateVirtualSpherePointSprite()
*
* This is simple example codes for point sprite test
*
***************************************************************/
MatrixTransform* ANIMCreateVirtualSpherePointSprite()
{
    MatrixTransform *psTrans = new MatrixTransform;

    Geode* ptGeode = new Geode();
    Geometry* ptGeometry = new Geometry();
    Vec3Array *vertices = new Vec3Array();
    Vec4Array *colors = new Vec4Array();

    vertices->push_back(Vec3(0, 0, 0));
    vertices->push_back(Vec3(0, 0, 1));
    colors->push_back(Vec4(1, 0, 0, 1));

    ptGeometry->setVertexArray(vertices);
    ptGeometry->setColorArray(colors);
    ptGeometry->setColorBinding(Geometry::BIND_OVERALL);

    DrawElementsUInt* points = new DrawElementsUInt(PrimitiveSet::POINTS, 0); 
    points->push_back(0);
    points->push_back(1);

    ptGeometry->addPrimitiveSet(points);
    ptGeode->addDrawable(ptGeometry);
    psTrans->addChild(ptGeode);

    /* create point sprite state set */
    StateSet *stateSet = new StateSet();

    stateSet->setMode(GL_BLEND, StateAttribute::ON);
    BlendFunc *func = new BlendFunc();
    func->setFunction(BlendFunc::SRC_ALPHA, BlendFunc::DST_ALPHA);
    stateSet->setAttributeAndModes(func, StateAttribute::ON);

    PointSprite *sprite = new PointSprite();
    stateSet->setTextureAttributeAndModes(0, sprite, StateAttribute::ON);

    Point *point = new Point();
    point->setSize(1000);
    stateSet->setAttribute(point);

    stateSet->setMode(GL_DEPTH_TEST, StateAttribute::OFF);
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    Texture2D *tex = new Texture2D();
    tex->setImage(osgDB::readImageFile(ANIMDataDir() + "Textures/ParticleFire.JPG"));
    stateSet->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);

    ptGeode->setStateSet(stateSet);

    return psTrans;
}


};






