#include "CudaTest.h"

#include <kernel/SceneManager.h>

#include <osg/Geode>

using namespace cvr;

CVRPLUGIN(CudaTest)

CudaTest::CudaTest()
{
}

CudaTest::~CudaTest()
{
}

bool CudaTest::init()
{
    _ctd = new CudaTestDrawable();

    osg::Geode * geode = new osg::Geode();

    geode->addDrawable(_ctd);

    SceneManager::instance()->getObjectsRoot()->addChild(geode);
    return true;
}
