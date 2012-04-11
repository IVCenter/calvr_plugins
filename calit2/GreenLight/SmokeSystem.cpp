#define ANIM_VIRTUAL_SPHERE_NUM_SAMPS 30

#include <osgParticle/Particle>
#include <osgParticle/ParticleSystem>
#include <osgParticle/ParticleSystemUpdater>
#include <osgParticle/ModularEmitter>
#include <osgParticle/ModularProgram>
#include <osgParticle/RandomRateCounter>
#include <osgParticle/SectorPlacer>
#include <osgParticle/RadialShooter>
#include <osgParticle/AccelOperator>

#include <stdio.h>
#include <string.h>

#include "GreenLight.h"


using namespace osg;
using namespace std;

MatrixTransform * GreenLight::InitSmoke()
{
    cout << "initializing smoke" << endl;

    MatrixTransform *psTrans = new MatrixTransform;
    MatrixTransform *psGreenflameTrans = new MatrixTransform;
    PositionAttitudeTransform* psRotTrans = new PositionAttitudeTransform;

    psRotTrans->addChild(psGreenflameTrans);

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

    osgParticle::Particle ptemplate1;
    ptemplate1.setLifeTime(2);
    ptemplate1.setShape(osgParticle::Particle::QUAD);
    ptemplate1.setSizeRange(osgParticle::rangef(0.f, 48.f));
    ptemplate1.setAlphaRange(osgParticle::rangef(0.4f, 0.6f));
    ptemplate1.setColorRange(osgParticle::rangev4(Vec4(0.2, 0.2, 0.2, 1.0), Vec4(0.2, 0.2, 0.2, 1.0)));
    ptemplate1.setRadius(0.5f);
    ptemplate1.setMass(0.05f);

    osgParticle::ParticleSystem *ps1 = new osgParticle::ParticleSystem;
    char particleTextureLocation[] = "/home/atarng/CALIT2/calvr_plugins/calit2/GreenLight";
    ps1->setDefaultAttributes( strcat(particleTextureLocation, "test-Particle.png"), true, false );
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

//    dsEmitterList->push_back(emitter1);
    return psTrans;
}
