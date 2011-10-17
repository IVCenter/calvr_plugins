#include "PanoViewLOD.h"

#include <kernel/NodeMask.h>
#include <kernel/PluginHelper.h>

#include <iostream>

using namespace cvr;

CVRPLUGIN(PanoViewLOD)

PanoViewLOD::PanoViewLOD()
{
}

PanoViewLOD::~PanoViewLOD()
{
}

bool PanoViewLOD::init()
{
    _root = new osg::MatrixTransform();
    _leftGeode = new osg::Geode();
    _rightGeode = new osg::Geode();

    _root->addChild(_leftGeode);
    _root->addChild(_rightGeode);

    _leftGeode->setNodeMask(_leftGeode->getNodeMask() & (~CULL_MASK_RIGHT));
    _rightGeode->setNodeMask(_rightGeode->getNodeMask() & (~CULL_MASK_LEFT));
    _rightGeode->setNodeMask(_rightGeode->getNodeMask() & (~CULL_MASK));


    _leftDrawable = new PanoDrawableLOD("/home/covise/data/PansLOD/LuxorTempleNight2-L_512_3.tif","/home/covise/data/PansLOD/LuxorTempleNight2-L_512_3.tif",9000,16,3,512);
    _rightDrawable = new PanoDrawableLOD("/home/covise/data/PansLOD/LuxorTempleNight2-L_512_3.tif","/home/covise/data/PansLOD/LuxorTempleNight2-L_512_3.tif",9000,16,3,512);

    _leftGeode->addDrawable(_leftDrawable);
    _rightGeode->addDrawable(_rightDrawable);


    PluginHelper::getScene()->addChild(_root);
}
