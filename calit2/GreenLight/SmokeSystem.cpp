#define ANIM_VIRTUAL_SPHERE_NUM_SAMPS 30


#include "GreenLight.h"

//CVRPLUGIN(GreenLight)

#include <stdio.h>
#include <string.h>
#include <cvrKernel/PluginHelper.h>

#include <osgParticle/Particle>
#include <osgParticle/ParticleSystem>
#include <osgParticle/ParticleSystemUpdater>
#include <osgParticle/ModularEmitter>
#include <osgParticle/ModularProgram>
#include <osgParticle/RandomRateCounter>
#include <osgParticle/SectorPlacer>
#include <osgParticle/RadialShooter>
#include <osgParticle/AccelOperator>
#include <osgParticle/MultiSegmentPlacer>
#include <osgParticle/FluidFrictionOperator>


using namespace osg;
using namespace std;

MatrixTransform * GreenLight::InitSmoke()
{
    cout << "initializing smoke" << endl;

    float pc[12];
    for ( int i = 0; i < 12; i++)
    {
        string p = "p";
        char numString[10];

        sprintf(numString, "%d", i);

        pc[i] = cvr::ConfigManager::getFloat( strcat( (char*) p.c_str(), numString),
                            "Plugin.GreenLight.ParticleTest", 0.0, NULL);
    }

    osg::Geode * testGeode = new osg::Geode;
//    osg::ShapeDrawable * testShape = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0,0,0),5,5,5));
    osg::MatrixTransform * test_MT = new MatrixTransform;
//    testGeode->addDrawable(testShape);
    osg::Vec3 testVector = osg::Vec3(pc[9],pc[10],pc[11]);
    osg::Matrixd testMatrix;
    testMatrix.makeScale(testVector);
    test_MT->setMatrix(testMatrix);
//    test_MT->addChild(testGeode);
    cvr::PluginHelper::getObjectsRoot()->addChild( test_MT );

    // Creates/Initializes ParticleSystem.
    // Sets Attributes of texture, emissive, and lighting.
    _osgParticleSystem = new osgParticle::ParticleSystem;
    _osgParticleSystem->setDefaultAttributes( "test-Particle.png", false, false );

    // PS is derived from Drawable, therefore we can create and add it as a child.
    osg::Geode * _particleGeode = new osg::Geode;
    cvr::PluginHelper::getObjectsRoot()->addChild( _particleGeode );
    _particleGeode->addDrawable( _osgParticleSystem );

    // adds an updater for per-frame management
    osgParticle::ParticleSystemUpdater * _osgPSUpdater = new osgParticle::ParticleSystemUpdater;
    _osgPSUpdater->addParticleSystem(_osgParticleSystem);
    cvr::PluginHelper::getObjectsRoot()->addChild( _osgPSUpdater );

    // Creates a particle to be used by the Particle System and define a few of its props.
    _pTemplate.setSizeRange(osgParticle::rangef(pc[0],pc[1])); // meters
    _pTemplate.setLifeTime(pc[2]);                             // seconds
    _pTemplate.setMass(pc[3]);                                 // kg
    _osgParticleSystem->setDefaultParticleTemplate(_pTemplate);

    /////////////////////////////////////

    // Creates a modular Emitter (has default counter, place and shooter.)
    osgParticle::ModularEmitter * emitter = new osgParticle::ModularEmitter;
    emitter->setParticleSystem(_osgParticleSystem);

    // rate at which new particles spawn
    osgParticle::RandomRateCounter * pRate =
        static_cast<osgParticle::RandomRateCounter *>(emitter->getCounter());
    pRate->setRateRange(pc[4],pc[5]);

    // customizes placer?
    osgParticle::MultiSegmentPlacer * lineSegment = new osgParticle::MultiSegmentPlacer();
    lineSegment->addVertex(0,0,-2);
    lineSegment->addVertex(0,-2,-2);
    lineSegment->addVertex(0,-16,0);
    emitter->setPlacer(lineSegment);

    // creates and initializes radial shooter.
    osgParticle::RadialShooter* smokeShooter = new osgParticle::RadialShooter();
    smokeShooter->setThetaRange(0.0, 3.14159/2); // radians relative to z axis.
    smokeShooter->setInitialSpeedRange(pc[6],pc[7]);  // meters/seconds.
    emitter->setShooter(smokeShooter);

    /////////////////////////////////////////////
    
    test_MT->addChild(emitter);

    //////////////////////////////////////////////

    osgParticle::ModularProgram * MPP = new osgParticle ::ModularProgram;
    MPP->setParticleSystem(_osgParticleSystem);

    osgParticle::AccelOperator * accelUp = new osgParticle::AccelOperator;
    accelUp->setToGravity(pc[8]);
    MPP->addOperator(accelUp);

    osgParticle::FluidFrictionOperator * airFriction = new osgParticle::FluidFrictionOperator;
    airFriction->setFluidToAir();
    MPP->addOperator(airFriction);

    cvr::PluginHelper::getObjectsRoot()->addChild( MPP );
    /***************************************/

    return test_MT; // probably unnecessary...
}
