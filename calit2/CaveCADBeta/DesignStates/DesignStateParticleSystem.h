/***************************************************************
* File Name: DesignStateParticleSystem.h
*
* Description: Particle system controller of DesignStateRenderer
*
***************************************************************/

#ifndef _DESIGN_STATE_PARTICLE_SYSTEM_H_
#define _DESIGN_STATE_PARTICLE_SYSTEM_H_


// Open scene graph
#include <osg/MatrixTransform>
#include <osgParticle/Emitter>


// Local include
#include "../AnimationModeler/ANIMDSParticleSystem.h"


/***************************************************************
* Class: DesignStateParticleSystem
***************************************************************/
class DesignStateParticleSystem: public osg::MatrixTransform
{
  public:
    DesignStateParticleSystem();

    void setEmitterEnabled(bool flag);

  protected:
    CAVEAnimationModeler::ANIMEmitterList mDSEmitterList;
};


#endif
