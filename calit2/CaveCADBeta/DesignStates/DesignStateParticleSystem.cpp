/***************************************************************
* File Name: DesignStateParticleSystem.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Oct 5, 2010
*
***************************************************************/
#include "DesignStateParticleSystem.h"

using namespace std;
using namespace osg;

//Constructor
DesignStateParticleSystem::DesignStateParticleSystem()
{
    mDSEmitterList.clear();
    addChild(CAVEAnimationModeler::ANIMCreateDesignStateParticleSystem(&mDSEmitterList));
}


/***************************************************************
* Function: setEmitterEnabled()
*
* Description: Enable/Disable all emitters of the system
*
***************************************************************/
void DesignStateParticleSystem::setEmitterEnabled(bool flag)
{
    if (mDSEmitterList.size() > 0)
    {
	CAVEAnimationModeler::ANIMEmitterList::iterator itrEmitter;
	for (itrEmitter = mDSEmitterList.begin(); itrEmitter != mDSEmitterList.end(); itrEmitter++)
	{
	    osgParticle::Emitter *emitterPtr = dynamic_cast <osgParticle::Emitter*> (*itrEmitter);
	    if (emitterPtr) emitterPtr->setEnabled(flag);
	}
    }
}



