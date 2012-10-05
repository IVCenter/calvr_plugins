#include "BinauralReality.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/InteractionManager.h>

BinauralReality * BinauralReality::_myPtr = NULL;

CVRPLUGIN(BinauralReality)

using namespace cvr;
using namespace osg;
using namespace std;

BinauralReality::BinauralReality()
{
    _myPtr = this;
}

BinauralReality::~BinauralReality()
{
}

BinauralReality * BinauralReality::instance()
{
    return _myPtr;
}

bool BinauralReality::init()
{
    _BinauralRealityMenu = new SubMenu("BinauralReality");

    PluginHelper::addRootMenuItem(_BinauralRealityMenu);

    return true;
}

void BinauralReality::menuCallback(MenuItem * item)
{

}

void BinauralReality::preFrame()
{

}

bool BinauralReality::processEvent(InteractionEvent * event)
{
    return false;
}

